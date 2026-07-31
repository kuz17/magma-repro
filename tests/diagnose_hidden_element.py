# tests/diagnose_hidden_element.py
"""
One-off diagnostic: print the FULL computed-style ancestor chain for a
known-phantom element, to find the actual CSS mechanism Amazon uses to
hide its collapsed category flyout menu. Run once, read the output, then
throw this away — not meant to be a permanent script.
"""
import sys
sys.path.insert(0, ".")
from src.agent.browser_env import BrowserEnv

with BrowserEnv(headless=True) as env:
    env.navigate("https://www.amazon.in/s?k=odyssey", wait="domcontentloaded")

    result = env._page.evaluate("""
        () => {
            // Find the phantom "Computers" link specifically
            const links = [...document.querySelectorAll('a')];
            const target = links.find(a => a.textContent.trim() === 'Computers');
            if (!target) return { error: 'element not found' };

            const chain = [];
            let node = target;
            while (node && node !== document.documentElement) {
                const s = window.getComputedStyle(node);
                chain.push({
                    tag: node.tagName,
                    class: node.className,
                    id: node.id,
                    display: s.display,
                    visibility: s.visibility,
                    opacity: s.opacity,
                    overflow: s.overflow,
                    overflowY: s.overflowY,
                    clientHeight: node.clientHeight,
                    clientWidth: node.clientWidth,
                    position: s.position,
                    clip: s.clip,
                    clipPath: s.clipPath,
                    transform: s.transform,
                    width: s.width,
                    height: s.height,
                    maxHeight: s.maxHeight,
                });
                node = node.parentElement;
            }
            return { chain };
        }
    """)

    import json
    print(json.dumps(result, indent=2))