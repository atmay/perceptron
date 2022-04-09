from PIL import Image, ImageDraw


class DotsDrawer:
    def __init__(self, filename, size=3):
        self.canvas_color = (255, 255, 255)
        self.canvas_size = (500, 500)
        self.filename = filename
        self.size = size

    def __enter__(self):
        self.image = Image.new(mode='RGB', size=self.canvas_size, color=self.canvas_color)
        self.draw = ImageDraw.Draw(self.image)
        return self

    def __exit__(self, type, value, traceback):
        with open(self.filename, 'wb') as out:
            self.image.save(out, 'PNG')
        self.image.close()

    def dot(self, x: int, y: int, color, size=None):
        size = size or self.size
        y = self.canvas_size[1] - y
        self.draw.arc(
            ((x - size, y - size), (x + size, y + size)),
            0, 360,
            fill=color,
            width=size * 2
        )