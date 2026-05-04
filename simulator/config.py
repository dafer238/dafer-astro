WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
VIEWPORT_WIDTH = 780
VIEWPORT_HEIGHT = 540

COLORS = {
    "bg": (10, 10, 26),
    "panel": (17, 17, 34),
    "frame": (26, 26, 46),
    "accent_blue": (0, 191, 255),
    "accent_orange": (255, 107, 53),
    "accent_green": (127, 255, 0),
    "accent_yellow": (255, 255, 0),
    "text": (200, 200, 200),
    "text_dim": (150, 150, 150),
    "orbit_color": (0, 191, 255),
    "satellite_color": (255, 255, 0),
    "earth_color": (0, 120, 200),
}

ORBIT_PRESETS = {
    "ISS": {"a": 6779.0, "e": 0.0001, "i": 51.6, "raan": 0.0, "omega": 0.0, "theta": 0.0, "n_orbits": 3.0},
    "GEO": {"a": 42157.0, "e": 0.0001, "i": 0.0, "raan": 0.0, "omega": 0.0, "theta": 0.0, "n_orbits": 1.0},
    "SSO": {"a": 6971.0, "e": 0.0001, "i": 97.8, "raan": 0.0, "omega": 0.0, "theta": 0.0, "n_orbits": 5.0},
}
