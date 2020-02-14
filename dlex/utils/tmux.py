import os
import subprocess
from typing import List


class TmuxManager:
    def __init__(self):
        self.panes = {}

    @property
    def current_pane_id(self):
        return os.getenv('TMUX_PANE')

    @property
    def is_tmux_session(self):
        return os.getenv('TMUX') is not None

    def split_window(self, pane_name, cmd="", height=10) -> int:
        if not self.is_tmux_session:
            return None
        pane_id = subprocess.check_output([
            "tmux", "split-window", "-l", str(height), "-P", "-F", "#{pane_id}", cmd
        ]).decode().strip()
        self.select_pane()
        self.panes[pane_name] = pane_id
        return pane_id

    def is_pane_visible(self, pane_name):
        return pane_name in self.panes

    def close_pane(self, pane_name):
        subprocess.call(["tmux", "kill-pane", "-t", self.panes[pane_name]])
        del self.panes[pane_name]

    def resize_pane(self, pane_name, height=None, width=None):
        if height:
            subprocess.call(["tmux", "resize-pane", "-t", self.panes[pane_name], "-y", str(height)])
        if width:
            subprocess.call(["tmux", "resize-pane", "-t", self.panes[pane_name], "-x", str(width)])

    def kill_pane(self, pand_id: str):
        subprocess.call(["tmux", "kill-pane", "-t", pand_id])

    def select_pane(self):
        if not self.is_tmux_session:
            return
        subprocess.call(["tmux", "select-pane", "-t", self.current_pane_id])

    def get_all_panes(self) -> List[str]:
        if not self.is_tmux_session:
            return []
        return subprocess.check_output(["tmux", "list-panes", "-F", "#{pane_id}"])\
            .decode().strip().split('\n')

    def close_all_panes(self):
        if not self.is_tmux_session:
            return
        for pane_id in self.get_all_panes():
            if pane_id != self.current_pane_id:
                subprocess.call(["tmux", "kill-pane", "-t", pane_id])


if __name__ == "__main__":
    tmux = TmuxManager()
    id = tmux.split_window("tail -f ")
    # tmux.kill_pane(id)
