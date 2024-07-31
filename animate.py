from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from common.constants import ORIGINAL_ACTION_SPACE
import re


def add_border_and_text(frame, step, env_idx, seed, taken_action, help_info, repeated_state):
    # help_info is dictionary containing `action_info`, list of (act, prob, logit); `entropy`; `distance`; and `need_help`
    border_size = 60
    # new_size = (frame.width + 2 * border_size, frame.height + border_size + border_size // 3)
    new_size = (frame.width + 3 * border_size, frame.height + border_size + border_size // 2)
    new_frame = Image.new("RGB", new_size, "white")
    new_frame.paste(frame, (border_size, border_size // 3))
    draw = ImageDraw.Draw(new_frame)
    # font = ImageFont.load_default()
    font = ImageFont.truetype("DejaVuSans.ttf", size = 7)
    draw.text((10, 10), f"Frame {step}", fill = "black", font = font)
    draw.text((10, 20), f"Env {env_idx}", fill = "black", font = font)
    draw.text((10, 30), f"Seed {seed}", fill = "black", font = font)

    taken_action = ORIGINAL_ACTION_SPACE[int(taken_action[0])]
    for action_info in help_info["action_info"]:
        if action_info[0] == taken_action:
            taken_action_prob = action_info[1]
            taken_action_logit = action_info[2]
            break
    if help_info["need_help"]:
        draw.text((10, new_size[1] - 60), f"{taken_action}", fill = "red", font = font)
    else:
        draw.text((10, new_size[1] - 60), f"{taken_action} | {taken_action_prob:.2f} | {taken_action_logit:.2f}", fill = "blue", font = font)
    taken_action_at_top = False
    for i, (action, prob, logit) in enumerate(sorted(help_info["action_info"], key = lambda t: t[1], reverse = True)[:5]):
        if action == taken_action:
            taken_action_at_top = True
        else:
            if i == 4:
                if taken_action_at_top:
                    draw.text((10, new_size[1] - 60 + (i + 1) * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)
            else:
                draw.text((10, new_size[1] - 60 + (i + 1) * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)
    # for i, (action, prob, logit) in enumerate(help_info["action_info"][5:10]):
    #     draw.text((100, new_size[1] - border_size + 85 + i * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)
    # for i, (action, prob, logit) in enumerate(help_info["action_info"][10:15]):
    #     draw.text((new_size[0] - border_size + 10, new_size[1] - border_size + 10 + i * 15), f"{action} | {prob:.2f} | {logit:.2f}", fill = "black", font = font)

    draw.text((new_size[0] - (2*border_size) + 10, 10), f"Entropy: {help_info['entropy']:.2f}", fill = "black", font = font)
    try:
        draw.text((new_size[0] - (2*border_size) + 10, 30), f"SVDD dist: {help_info['distance']:.6f}", fill = "black", font = font)
    except KeyError:  # not doing svdd => no distance
        pass
    if help_info["need_help"]:
        draw.text((new_size[0] - border_size + 10, 50), "Asked for help!!!", fill = "red", font = font)
    # if repeated_state:
        # draw.text((new_size[0] - border_size + 10, 70), "Repeated state!", fill = "green", font = font)
    
    return new_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type = str, required = True)
    # parser.add_argument("--env", "-e", type = int, nargs = "+", required = True)  # quant eval's environment index
    # parser.add_argument("--seed", "-s", type = int, nargs = "+", required = True)  # seed associated with environment
    parser.add_argument("--display", "-p", action = "store_true", default = False)
    args = parser.parse_args()

    frame_duration = 0.25
    all_logs = os.listdir(args.dir)
    for i, log_file in enumerate(all_logs):
        if log_file.startswith("AAA_storage_") and log_file.endswith(".pkl"):
            env_idx = int(re.search(r"env_(\d+)", log_file).group(1))
            seed = int(re.search(r"seed_(\d+)", log_file).group(1))
            with open(os.path.join(args.dir, log_file), "rb") as f:
                run_info = pickle.load(f)
            all_action_info = run_info["help_info_storage"]
            taken_actions = run_info["action_storage"]
            repeated_states_log = run_info["repeated_state_storage"]
            step = 0
            frames = []
            while True:
                try:
                    img_path = os.path.join(args.dir, f"obs_env_{env_idx}_seed_{seed}_step_{step}.png")
                    frame = Image.open(img_path)
                    frame = add_border_and_text(frame, step, env_idx, seed, taken_actions[step], all_action_info[step], repeated_states_log[step])
                    frames.append(frame)
                    step += 1
                except (IndexError, FileNotFoundError):
                    break
            num_frames = len(frames)

            # Save
            gif_name = f"AAA_env_{env_idx}_seed_{seed}_trajectory.gif"
            frames[0].save(os.path.join(args.dir, gif_name), save_all = True, append_images = frames[1:], duration = frame_duration * 1000, loop = 0)
            print(gif_name, "saved")

            # Display
            if args.display:
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
