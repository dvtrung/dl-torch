import curses
import subprocess

from typing import Dict


class CursesManager:
    def __init__(self):
        self.main_screen = None
        self.main_text = ""
        self.modal_screen = None

    def wrapper(self, func):
        def _wrapper(scr):
            scr.nodelay(True)
            curses.curs_set(0)
            self.main_screen = scr
            func(scr)
        curses.wrapper(_wrapper)

    def get_modal_window(self, height, width):
        maxy, maxx = self.main_screen.getmaxyx()
        return curses.newwin(height, width, (maxy - height) // 2, (maxx - width) // 2)

    @property
    def height(self):
        return self.main_screen.getmaxyx()[0]

    @property
    def width(self):
        return self.main_screen.getmaxyx()[1]

    def select(self, choices: Dict[str, str], default, title="Menu") -> str:
        idx_to_key = [key for i, key in enumerate(choices.keys())]
        scr_width = max([len(c) for c in choices.values()] + [len(title)]) + 10
        scr_height = len(choices) + 4
        screen = self.get_modal_window(scr_height, scr_width)
        self.modal_screen = screen
        selection = -1
        option = 0
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        while selection < 0:
            # screen.clear()
            screen.box()
            h = [0] * len(choices)  # curses.A_NORMAL = 0
            h[option] = curses.color_pair(1)
            # h[option] = curses.A_REVERSE
            screen.addstr(0, 2, f" {title} ")

            for i, (key, s) in enumerate(choices.items()):
                screen.addstr(2 + i, 4, s, h[i])

            self.main_screen.refresh()
            screen.refresh()

            q = self.main_screen.getch()
            if q == curses.KEY_UP or q == ord('k'):  # KEY_UP or 'k' on vim mode
                option = (option - 1) % len(choices)
            elif q == curses.KEY_DOWN or q == ord('j'):  # KEY_DOWN or 'j' on vim mode
                option = (option + 1) % len(choices)
            elif q == ord('\n'):
                selection = option
            elif q == ord('m'):
                selection = None
                break

        del self.modal_screen
        self.modal_screen = None
        self.main_screen.refresh()
        return idx_to_key[selection] if selection is not None else None

    def refresh(self, clear=False):
        try:
            # os.system('clear')
            if clear:
                self.main_screen.clear()
            self.main_screen.move(0, 0)
            self.main_screen.addstr(0, 0, self.main_text)
        except Exception:
            pass
        finally:
            pass
            # self.main_screen.refresh()
            # if self.modal_screen:
            #     self.modal_screen.refresh()
            # self.main_screen.refresh()
            # self.main_screen.move(curses.LINES - 1, 0)

    def show_info_dialog(self, cmd, interval=1):
        assert self.modal_screen is None
        screen = self.main_screen
        # screen.box()
        # self.modal_screen = screen
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        self.main_screen.clear()
        self.modal_screen.clear()

        key_pressed = None
        lines = []
        while key_pressed not in [ord("x"), ord("q")]:
            key_pressed = self.main_screen.getch()
            text = self.proc.stdout.readline()
            lines += [text[i:i+self.width] for i in range(len(text) // self.width + 1)]
            for i, line in enumerate(lines[-self.height:]):
                screen.addstr(i, 0, line)
            screen.refresh()
            # self.main_screen.refresh()
        self.main_screen.clear()
        self.refresh()