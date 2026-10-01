"""Rendering: draw speed frames, export PNGs and movies.

Single responsibility: turning speed fields into pictures. ``FrameArtist``
draws one frame; ``PngExporter`` / ``MovieExporter`` write files. No physics,
no simulation state -- they only see numpy arrays and polygons.
"""

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib import animation


class FrameArtist:
    """Draws one speed-magnitude frame with obstacle outlines on axes."""

    def __init__(self, polygons, lx: float, ly: float):
        self.polygons = polygons
        self.lx = lx
        self.ly = ly

    def draw(self, ax, speed, title, vmin=0.0, vmax=None):
        """Render onto ``ax``; returns the image (for colorbars)."""
        ax.clear()
        im = ax.imshow(
            speed, origin="lower", cmap="turbo",
            extent=[0, self.lx, 0, self.ly], aspect="auto",
            vmin=vmin, vmax=vmax,
        )
        for poly in self.polygons:
            pts = poly.closed_points()
            ax.fill(pts[:, 0], pts[:, 1], color="white", ec="black", lw=1.5, zorder=3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        return im


class PngExporter:
    """Writes the final speed frame to a PNG file."""

    def __init__(self, artist: FrameArtist):
        self.artist = artist

    def save(self, path, speed, title, vmax=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        im = self.artist.draw(ax, speed, title, vmax=vmax)
        fig.colorbar(im, ax=ax, label="m/s")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"wrote image: {path}")


class MovieExporter:
    """Writes recorded speed frames to a .mov file (ffmpeg + H.264)."""

    def __init__(self, artist: FrameArtist):
        self.artist = artist

    def save(self, path, frames, *, fps, dt, stride, vmax=None):
        if shutil.which("ffmpeg") is None:
            print("warning: ffmpeg not found, skipping movie", file=sys.stderr)
            return
        if not frames:
            print("warning: no frames recorded, skipping movie", file=sys.stderr)
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        im = self.artist.draw(ax, frames[0],
                              f"speed, step {stride} (t={stride * dt:.3f} s)",
                              vmax=vmax)
        fig.colorbar(im, ax=ax, label="m/s")

        def update(i):
            s = i * stride
            self.artist.draw(ax, frames[i],
                             f"speed, step {s} (t={s * dt:.3f} s)",
                             vmax=vmax)

        ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                      interval=1000 // fps)
        # yuv420p: QuickTime Player only accepts H.264 with 4:2:0 chroma
        writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
        ani.save(path, writer=writer)
        plt.close(fig)
        print(f"wrote movie: {path} "
              f"({len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")
