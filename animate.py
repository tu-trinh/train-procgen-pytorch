from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type = str, required = True)
    parser.add_argument("--env", "-e", type = int, default = 1)
    parser.add_argument("--epoch", "-p", type = int, default = 1)
    args = parser.parse_args()

    frame_duration = 0.25
    for env in range(args.env):
        for epoch in range(args.epoch):
            step = 0
            frames = []
            while True:
                try:
                    img_path = os.path.join(args.dir, f"obs_env_{env}_epoch_{epoch}_step_{step}.png")
                    frame = Image.open(img_path)
                except:
                    break
                ImageDraw.Draw(frame).text((10, 10), f"Frame {step}", font = ImageFont.load_default())
                frames.append(frame)
                step += 1
            num_frames = len(frames)

            # Save
            frames[0].save(os.path.join(args.dir, f"env_{env}_epoch_{epoch}_trajectory.gif"), save_all = True, append_images = frames[1:], duration = frame_duration * 1000, loop = 0)

            # Display
            fig = plt.figure()
            plt.axis("off")
            image = plt.imshow(frames[0])
            frame_idx = 0
            is_paused = False
            
            def animate(i):
                image.set_data(frames[i])
                return [image]
            
            anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval = frame_duration * 1000, blit = True, repeat = False)
            
            def on_press(event):
                global frame_idx, is_paused
                if event.key == " ":
                    if is_paused:
                        anim.resume()
                    else:
                        anim.pause()
                    is_paused = not is_paused
            fig.canvas.mpl_connect("key_press_event", on_press)
            plt.show()
