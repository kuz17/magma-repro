from PIL import ImageDraw

from src.utils.bbox import denormalize_bbox


def apply_som(image, elements):

    draw = ImageDraw.Draw(image)

    width, height = image.size

    for idx, element in enumerate(elements):

        bbox = element["bbox"]

        x1, y1, x2, y2 = denormalize_bbox(
            bbox,
            width,
            height,
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2,
        )

        draw.text(
            (x1, y1),
            str(idx),
            fill="red",
        )

    return image
