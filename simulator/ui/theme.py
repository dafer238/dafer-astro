import dearpygui.dearpygui as dpg


def create_theme():
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 10, 26))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (17, 17, 34))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (17, 17, 34))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (26, 26, 46))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (36, 36, 60))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (46, 46, 76))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (17, 17, 34))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (26, 26, 46))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (10, 10, 26))
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (17, 17, 34))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (10, 10, 26))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (50, 50, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (70, 70, 110))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 191, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 191, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (0, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (36, 36, 60))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 120, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 160, 230))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (26, 26, 46))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (36, 36, 60))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (46, 46, 76))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (50, 50, 80))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200))
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, (0, 191, 255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 6, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 3)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4)

    return global_theme


def create_button_themes():
    """Create themed button styles for different action types."""
    # Green: add/create actions
    with dpg.theme() as green_btn:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 60, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 100, 45))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 130, 60))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (140, 255, 140))

    # Red: delete/clear actions
    with dpg.theme() as red_btn:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 20, 20))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 30, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (130, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 140, 140))

    # Yellow: update/modify actions
    with dpg.theme() as yellow_btn:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 45, 15))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 70, 20))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (110, 95, 30))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 240, 140))

    # White: compute/primary action
    with dpg.theme() as white_btn:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 200, 210))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 230, 240))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 20))

    return {"green": green_btn, "red": red_btn, "yellow": yellow_btn, "white": white_btn}
