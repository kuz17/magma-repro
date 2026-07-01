# src/agent/browser_env.py
"""
Playwright browser environment for the magma-repro web agent.

Owns the browser process; exposes a clean interface for screenshot, click,
navigate, scroll, and type.  All click/scroll coordinates are expressed as
normalised floats in [0, 1] so callers never deal with raw pixels — the
viewport-to-pixel conversion lives entirely here.

Usage (standalone smoke-test):
    python -m src.agent.browser_env --url https://www.google.com

Dependencies:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)


# ── tunables ───────────────────────────────────────────────────────────────
DEFAULT_VIEWPORT_W = 1280
DEFAULT_VIEWPORT_H = 800
STABILITY_TIMEOUT_MS = 5_000   # wait_for_load_state timeout
STABILITY_POLL_MS    = 200     # polling interval for custom stability check
STABILITY_MAX_DIFF   = 0.02    # max mean-pixel-diff (0–1) to consider stable
STABILITY_RETRIES    = 10      # how many polls before giving up
SCREENSHOT_DIR       = Path("outputs/browser")  # for debug saves


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreenshotResult:
    """Return value of BrowserEnv.screenshot()."""
    image:         Image.Image  # PIL image, CSS-pixel resolution
    png_bytes:     bytes        # raw PNG (for sending to inference server)
    viewport_w:    int          # CSS-pixel width  (= image.width  when DPR=1)
    viewport_h:    int          # CSS-pixel height (= image.height when DPR=1)
    dpr:           float        # device pixel ratio (usually 1.0 in headless)
    # dpr_x / dpr_y let callers detect any DPR mismatch:
    dpr_x:         float        # png_width  / viewport_w
    dpr_y:         float        # png_height / viewport_h


@dataclass
class ClickResult:
    """Return value of BrowserEnv.click()."""
    x_viewport:  float   # CSS-pixel x actually clicked
    y_viewport:  float   # CSS-pixel y actually clicked
    x_norm:      float   # normalised input
    y_norm:      float   # normalised input
    changed:     bool    # did the page visually change after the click?


# ══════════════════════════════════════════════════════════════════════════════
# BrowserEnv
# ══════════════════════════════════════════════════════════════════════════════

class BrowserEnv:
    """
    Thin wrapper around a Playwright Chromium page.

    Coordinate contract
    ───────────────────
    All public methods that take positional arguments use NORMALISED
    coordinates (x_norm, y_norm) ∈ [0, 1], measured over the CSS-pixel
    viewport.  Internally they are converted to viewport pixels via:

        px_x = x_norm * viewport_w
        px_y = y_norm * viewport_h

    This matches the normalised output of OmniParser / DemoRunner exactly,
    so the caller never needs to touch pixel arithmetic.

    DPR handling
    ────────────
    Playwright's page.screenshot() with the default scale='css' returns an
    image whose pixel dimensions equal the CSS viewport dimensions when
    DPR=1.  On some headless environments Chromium sets DPR=2; in that case
    the PNG is 2× larger than the viewport in each axis.

    __init__ detects this and stores dpr_x / dpr_y.  click() always converts
    through viewport coordinates (not PNG pixels), so it remains correct
    regardless of DPR.  A warning is emitted if DPR ≠ 1 so you notice.
    """

    def __init__(
        self,
        headless: bool = True,
        viewport_w: int = DEFAULT_VIEWPORT_W,
        viewport_h: int = DEFAULT_VIEWPORT_H,
        slow_mo_ms: int = 0,
        save_screenshots: bool = False,
    ):
        """
        Launch the browser.  Call .close() when done, or use as a context manager.

        Args:
            headless:          Run without a visible window (True for agent use).
            viewport_w/h:      CSS-pixel viewport size.
            slow_mo_ms:        Slow Playwright actions by N ms (useful for debugging).
            save_screenshots:  If True, every screenshot() call saves a PNG to
                               SCREENSHOT_DIR.  Off by default to avoid disk churn.
        """
        from playwright.sync_api import sync_playwright

        self._pw        = sync_playwright().__enter__()
        self._browser   = self._pw.firefox.launch(
            headless=headless,
            slow_mo=slow_mo_ms,
        )
        self._context   = self._browser.new_context(
            viewport={"width": viewport_w, "height": viewport_h},
        )
        self._page      = self._context.new_page()
        self._vp_w      = viewport_w
        self._vp_h      = viewport_h
        self._save_ss   = save_screenshots
        self._ss_count  = 0

        # Detect DPR once after launch via JS
        self._dpr = float(self._page.evaluate("window.devicePixelRatio") or 1.0)
        if self._dpr != 1.0:
            log.warning(
                "Device pixel ratio = %.1f (not 1.0). "
                "PNG screenshots will be %dx the CSS viewport size. "
                "Coordinate remapping will compensate automatically.",
                self._dpr, int(self._dpr),
            )

        if save_screenshots:
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        log.info(
            "BrowserEnv ready: viewport=%dx%d  dpr=%.1f  headless=%s",
            viewport_w, viewport_h, self._dpr, headless,
        )

    # ── navigation ────────────────────────────────────────────────────────

    def navigate(self, url: str, wait: str = "networkidle") -> None:
        """
        Navigate to url and wait for the page to settle.

        Args:
            url:  Full URL including scheme (e.g. "https://google.com").
            wait: Playwright load state — "load" | "domcontentloaded" |
                  "networkidle".  "networkidle" is safest for SPAs.
        """
        log.debug("navigate → %s", url)
        self._page.goto(url, wait_until=wait, timeout=30_000)
        log.info("navigated to %s", url)

    # ── screenshot ────────────────────────────────────────────────────────

    def screenshot(self, wait_stable: bool = True) -> ScreenshotResult:
        """
        Capture the current viewport as a PIL image + PNG bytes.

        Args:
            wait_stable: If True, poll until the page stops changing visually
                         before capturing.  Adds up to
                         STABILITY_RETRIES × STABILITY_POLL_MS ms of latency
                         but prevents capturing a mid-render frame.

        Returns:
            ScreenshotResult with the image, raw bytes, and coordinate metadata.
        """
        if wait_stable:
            self._wait_for_visual_stability()

        raw: bytes = self._page.screenshot(type="png")  # scale='css' is default
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Compute DPR from actual PNG dimensions vs reported viewport
        # (re-querying viewport_size() here rather than using __init__ values
        #  because the user might have resized the window between calls)
        vp = self._page.viewport_size or {"width": self._vp_w, "height": self._vp_h}
        vp_w, vp_h = vp["width"], vp["height"]
        dpr_x = img.width  / vp_w
        dpr_y = img.height / vp_h

        if abs(dpr_x - 1.0) > 0.01 or abs(dpr_y - 1.0) > 0.01:
            log.warning(
                "DPR mismatch: PNG is %dx%d but viewport reports %dx%d  "
                "(dpr_x=%.2f dpr_y=%.2f).  Clicks will be compensated.",
                img.width, img.height, vp_w, vp_h, dpr_x, dpr_y,
            )

        result = ScreenshotResult(
            image=img,
            png_bytes=raw,
            viewport_w=vp_w,
            viewport_h=vp_h,
            dpr=self._dpr,
            dpr_x=dpr_x,
            dpr_y=dpr_y,
        )

        if self._save_ss:
            path = SCREENSHOT_DIR / f"ss_{self._ss_count:04d}.png"
            img.save(path)
            log.debug("screenshot saved → %s", path)
            self._ss_count += 1

        return result

    # ── actions ───────────────────────────────────────────────────────────

    def click(
        self,
        x_norm: float,
        y_norm: float,
        ss: Optional[ScreenshotResult] = None,
        verify_change: bool = False,
    ) -> ClickResult:
        """
        Click at a normalised position.

        Args:
            x_norm, y_norm: Position ∈ [0, 1] over the CSS viewport.
                            These should come directly from DemoRunner /
                            inference server output — no scaling needed.
            ss:             The ScreenshotResult that produced these coords.
                            If supplied, dpr_x/dpr_y are taken from it
                            (handles per-call viewport resizes gracefully).
                            If None, the class-level DPR is used.
            verify_change:  If True, take a before/after screenshot and report
                            whether the page visually changed.  Adds ~0.5s.

        Returns:
            ClickResult with the actual CSS-pixel coordinates used.
        """
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise ValueError(
                f"Normalised coords must be in [0,1]: got ({x_norm:.4f}, {y_norm:.4f})"
            )

        # Use viewport size from the screenshot that produced these coords,
        # falling back to the live viewport.
        if ss is not None:
            vp_w, vp_h = ss.viewport_w, ss.viewport_h
        else:
            vp = self._page.viewport_size or {"width": self._vp_w, "height": self._vp_h}
            vp_w, vp_h = vp["width"], vp["height"]

        # Convert to CSS-pixel coordinates.
        # We multiply by the VIEWPORT size (not PNG size) because
        # page.mouse.click() takes CSS pixels, not physical pixels.
        px_x = x_norm * vp_w
        px_y = y_norm * vp_h

        log.debug(
            "click  norm=(%.4f, %.4f)  viewport_px=(%.1f, %.1f)  vp=%dx%d",
            x_norm, y_norm, px_x, px_y, vp_w, vp_h,
        )

        if verify_change:
            pre_raw = self._page.screenshot(type="png")

        self._page.mouse.click(px_x, px_y)

        changed = False
        if verify_change:
            time.sleep(0.4)   # brief settle before comparison screenshot
            post_raw = self._page.screenshot(type="png")
            changed = _images_differ(pre_raw, post_raw)
            if not changed:
                log.warning(
                    "Click at (%.1f, %.1f) produced no visual change — "
                    "possible miss or already-selected element.",
                    px_x, px_y,
                )

        return ClickResult(
            x_viewport=px_x,
            y_viewport=px_y,
            x_norm=x_norm,
            y_norm=y_norm,
            changed=changed,
        )

    def type_text(self, text: str, delay_ms: int = 50) -> None:
        """
        Type text at the currently focused element.

        Args:
            text:     The string to type.
            delay_ms: Delay between keystrokes in ms; 0 = instant.
        """
        log.debug("type: %r", text)
        self._page.keyboard.type(text, delay=delay_ms)

    def press_key(self, key: str) -> None:
        """
        Press a single key by name (e.g. "Enter", "Tab", "Escape").
        Playwright key names: https://playwright.dev/python/docs/api/class-keyboard
        """
        log.debug("press: %r", key)
        self._page.keyboard.press(key)

    def scroll(
        self,
        x_norm: float = 0.5,
        y_norm: float = 0.5,
        delta_y: int = 300,
    ) -> None:
        """
        Scroll the page at a normalised position.

        Args:
            x_norm, y_norm: Scroll origin (default: centre of viewport).
            delta_y:        Pixel delta; positive = scroll down.
        """
        vp = self._page.viewport_size or {"width": self._vp_w, "height": self._vp_h}
        px_x = x_norm * vp["width"]
        px_y = y_norm * vp["height"]
        log.debug("scroll at (%.1f, %.1f) delta_y=%d", px_x, px_y, delta_y)
        self._page.mouse.wheel(delta_x=0, delta_y=delta_y)

    def get_interactive_elements(self) -> list[dict]:
        """
        Extract visible interactive DOM elements via JavaScript.

        Queries <input>, <textarea>, <select>, <button>, and ARIA equivalents.
        Returns a list of dicts: {tag, type, label, bbox_norm: [x1,y1,x2,y2]}
        where bbox_norm coordinates are normalised to [0,1] over the viewport.
        Elements outside the visible viewport or with negligible size are skipped.
        """
        vp = self._page.viewport_size or {"width": self._vp_w, "height": self._vp_h}
        vp_w, vp_h = vp["width"], vp["height"]

        raw = self._page.evaluate("""
            () => {
                const sel = 'input:not([type="hidden"]), textarea, select, button, '
                          + '[role="searchbox"], [role="combobox"], [role="textbox"]';
                const out = [];
                for (const el of document.querySelectorAll(sel)) {
                    const r = el.getBoundingClientRect();
                    if (r.width < 5 || r.height < 5) continue;
                    if (r.bottom < 0 || r.top > window.innerHeight) continue;
                    if (r.right  < 0 || r.left > window.innerWidth)  continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden'
                            || style.opacity === '0') continue;
                    out.push({
                        tag:  el.tagName.toLowerCase(),
                        type: el.type || '',
                        placeholder: el.placeholder || el.getAttribute('aria-label')
                                     || el.getAttribute('aria-placeholder') || '',
                        name: el.name || el.id || '',
                        x1: r.left, y1: r.top, x2: r.right, y2: r.bottom,
                    });
                }
                return out;
            }
        """)

        elements = []
        for e in raw or []:
            x1 = max(0.0, e["x1"] / vp_w)
            y1 = max(0.0, e["y1"] / vp_h)
            x2 = min(1.0, e["x2"] / vp_w)
            y2 = min(1.0, e["y2"] / vp_h)
            if x2 - x1 < 0.01 or y2 - y1 < 0.005:
                continue
            label = e["placeholder"] or e["name"] or f"{e['tag']}[{e['type']}]"
            elements.append({
                "tag":      e["tag"],
                "type":     e["type"],
                "label":    label,
                "bbox_norm": [x1, y1, x2, y2],
            })
        log.debug("get_interactive_elements: %d elements found", len(elements))
        return elements

    def wait_for_load(self, state: str = "networkidle", timeout_ms: int = 10_000) -> None:
        """
        Block until the page reaches a load state.

        Args:
            state:      "load" | "domcontentloaded" | "networkidle"
            timeout_ms: Maximum wait in ms.
        """
        self._page.wait_for_load_state(state, timeout=timeout_ms)

    def current_url(self) -> str:
        return self._page.url

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "BrowserEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the browser and Playwright handle."""
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw.__exit__(None, None, None)
        except Exception:
            pass
        log.info("BrowserEnv closed.")

    # ── internal ──────────────────────────────────────────────────────────

    def _wait_for_visual_stability(self) -> None:
        """
        Poll screenshots until two consecutive frames are visually similar.

        Uses a fast mean-absolute-difference check on downsampled images to
        avoid the overhead of comparing full-resolution PNGs.  Falls back
        after STABILITY_RETRIES polls regardless.
        """
        # First wait for Playwright's built-in network idle signal
        try:
            self._page.wait_for_load_state(
                "networkidle", timeout=STABILITY_TIMEOUT_MS
            )
        except Exception:
            pass   # timeout is fine — we'll still do the pixel check

        # Then do a pixel-level stability check to catch JS-driven renders
        prev_raw: Optional[bytes] = None
        for attempt in range(STABILITY_RETRIES):
            curr_raw = self._page.screenshot(type="png")
            if prev_raw is not None:
                if not _images_differ(prev_raw, curr_raw, threshold=STABILITY_MAX_DIFF):
                    log.debug("visual stability reached after %d polls", attempt + 1)
                    return
            prev_raw = curr_raw
            time.sleep(STABILITY_POLL_MS / 1000.0)

        log.debug("stability timeout after %d polls — proceeding anyway", STABILITY_RETRIES)


# ── helpers ────────────────────────────────────────────────────────────────

def _images_differ(
    raw_a: bytes,
    raw_b: bytes,
    thumbnail_size: tuple = (160, 100),
    threshold: float = STABILITY_MAX_DIFF,
) -> bool:
    """
    Return True if two PNG byte-strings differ by more than threshold.

    Downsamples to thumbnail_size first for speed.  threshold is mean
    absolute pixel difference normalised to [0, 1].
    """
    import numpy as np

    img_a = Image.open(io.BytesIO(raw_a)).convert("L").resize(thumbnail_size)
    img_b = Image.open(io.BytesIO(raw_b)).convert("L").resize(thumbnail_size)
    arr_a = np.array(img_a, dtype=float)
    arr_b = np.array(img_b, dtype=float)
    diff  = float(np.mean(np.abs(arr_a - arr_b))) / 255.0
    return diff > threshold


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test(url: str, x_norm: float, y_norm: float, save: bool, headless: bool = False) -> None:
    """
    Navigate to url, screenshot, click at (x_norm, y_norm), screenshot again.
    Verifies coordinate math and DPR handling without any model involvement.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"\n{'─'*60}")
    print(f"  BrowserEnv smoke test")
    print(f"  url     : {url}")
    print(f"  target  : ({x_norm}, {y_norm})  [normalised]")
    print(f"{'─'*60}\n")

    with BrowserEnv(headless=headless, save_screenshots=save) as env:
        env.navigate(url)

        ss = env.screenshot(wait_stable=True)
        print(f"Screenshot:")
        print(f"  PNG size      : {ss.image.width} × {ss.image.height} px")
        print(f"  Viewport size : {ss.viewport_w} × {ss.viewport_h} css-px")
        print(f"  DPR           : {ss.dpr:.2f}  (dpr_x={ss.dpr_x:.2f} dpr_y={ss.dpr_y:.2f})")

        # Assert coordinate sanity
        expected_px_x = x_norm * ss.viewport_w
        expected_px_y = y_norm * ss.viewport_h
        print(f"\nExpected click (css-px): ({expected_px_x:.1f}, {expected_px_y:.1f})")

        result = env.click(x_norm, y_norm, ss=ss, verify_change=True)
        print(f"\nClick result:")
        print(f"  css-px        : ({result.x_viewport:.1f}, {result.y_viewport:.1f})")
        print(f"  visual change : {result.changed}")

        assert abs(result.x_viewport - expected_px_x) < 0.5, "x coordinate mismatch"
        assert abs(result.y_viewport - expected_px_y) < 0.5, "y coordinate mismatch"

        print(f"\n✓  Coordinate math correct.")

        if save:
            print(f"  Screenshots saved to {SCREENSHOT_DIR}/")

        # Also verify the screenshot-to-click viewport contract explicitly
        ss2 = env.screenshot()
        print(f"\nPost-click screenshot size: {ss2.image.width} × {ss2.image.height}")
        print(f"Current URL : {env.current_url()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowserEnv smoke test")
    parser.add_argument("--url",      default="https://www.google.com",
                        help="URL to navigate to")
    parser.add_argument("--x",        type=float, default=0.5,
                        help="Normalised x click target [0-1]")
    parser.add_argument("--y",        type=float, default=0.5,
                        help="Normalised y click target [0-1]")
    parser.add_argument("--save",     action="store_true",
                        help=f"Save screenshots to {SCREENSHOT_DIR}/")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run headless (no visible window). Default: False")
    args = parser.parse_args()
    _smoke_test(args.url, args.x, args.y, args.save, args.headless)


if __name__ == "__main__":
    main()