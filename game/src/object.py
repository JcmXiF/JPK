# Object类：所有游戏实体的基类，负责从配置加载图像（带全局缓存）、帧动画和碰撞矩形计算。
import pygame

from game.src.const import objectdata

_image_cache = {}


class Object(pygame.sprite.Sprite):
    def __init__(self, dataname, id):
        super().__init__()
        self.dataname = dataname
        self.id = id

        self.data = self.obtaindata()
        self.name = self.data["name"]
        self.path = self.data["path"]
        self.frames = self.data["frames"]
        self.size = self.data["size"]
        self.speed = self.data["speed"]
        self.colorkey = self.data["colorkey"]

        self.images = self._load_all_frames()
        self.image = self.images[0]

        self._half_w = self.size[0] // 2
        self._half_h = self.size[1] // 2

        self.frame_index = 0
        self._last_frame_time = 0
        self._frame_duration = 500

    def _load_all_frames(self):
        """加载所有动画帧到缓存并返回帧列表。"""
        loaded = []
        for i in range(self.frames):
            path = self.path % i
            if path not in _image_cache:
                try:
                    img = pygame.image.load(path).convert_alpha()
                except pygame.error:
                    print(f"Can't load image: {path}")
                    raise SystemExit(pygame.error)

                if self.colorkey is not None:
                    ck = self.colorkey
                    if ck == -1:
                        ck = img.get_at((0, 0))
                    img.set_colorkey(ck)
                _image_cache[path] = img
            loaded.append(_image_cache[path])
        return loaded

    def obtaindata(self):
        return self.dataname[self.id]

    def draw(self, sc):
        self.frame_animation()
        sc.blit(self.image, (self.position[0] - self._half_w,
                             self.position[1] - self._half_h))

    def frame_animation(self):
        if self.frames <= 1:
            return
        now = pygame.time.get_ticks()
        if now - self._last_frame_time >= self._frame_duration:
            self._last_frame_time = now
            self.frame_index = (self.frame_index + 1) % self.frames
            self.image = self.images[self.frame_index]

    def calculate_rect(self, position, size):
        rect = pygame.Rect(0, 0, size[0], size[1])
        rect.center = position
        return rect
