from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


def add_border_and_text(frame, step, help_info):
    # help_info is dictionary containing `action_info` list of (act, prob, logit), `entropy`, and `need_help`
    border_size = 100
    # new_size = (frame.width + 2 * border_size, frame.height + border_size + border_size // 3)
    new_size = (frame.width + 2 * border_size, frame.height + border_size + border_size // 3)
    new_frame = Image.new("RGB", new_size, "white")
    new_frame.paste(frame, (border_size, border_size // 3))
    draw = ImageDraw.Draw(new_frame)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Frame {step}", fill = "black", font = font)

    for i, (action, prob, logit) in enumerate(help_info["action_info"][:5]):
        fill = "blue" if i == 0 else "black"
        draw.text((10, new_size[1] - border_size + 10 + i * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = fill, font = font)
    # for i, (action, prob, logit) in enumerate(help_info["action_info"][5:10]):
    #     draw.text((100, new_size[1] - border_size + 85 + i * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)
    # for i, (action, prob, logit) in enumerate(help_info["action_info"][10:15]):
    #     draw.text((new_size[0] - border_size + 10, new_size[1] - border_size + 10 + i * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)

    draw.text((new_size[0] - border_size + 10, 10), f"Entropy: {help_info['entropy']:.2f}", fill = "black", font = font)
    if help_info["need_help"]:
        draw.text((new_size[0] - border_size + 10, 30), "Asked for help!!!", fill = "red", font = font)
    
    return new_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type = str, required = True)
    parser.add_argument("--env", "-e", type = int, default = 1)
    parser.add_argument("--epoch", "-p", type = int, default = 1)
    args = parser.parse_args()

    frame_duration = 0.25
    for env in range(args.env):
        for epoch in range(args.epoch):
            with open(os.path.join(args.dir, f"storage_epoch_{epoch}.pkl"), "rb") as f:
                all_action_info = pickle.load(f)
            step = 0
            frames = []
            while True:
                try:
                    img_path = os.path.join(args.dir, f"obs_env_{env}_epoch_{epoch}_step_{step}.png")
                    frame = Image.open(img_path)
                    frame = add_border_and_text(frame, step, all_action_info[step])
                    frames.append(frame)
                    step += 1
                except:
                    break
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
