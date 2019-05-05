import sys
import glob

def show_menu(options):
    for i, opt in enumerate(options):
        print("%d) %s" % (i + 1, opt))

    prompt = "Choose an action: "
    ret = input(prompt)
    while not ret.isdigit() or int(ret) <= 0 or int(ret) > len(options):
        ret = input(prompt)
    return ret

if sys.argv[1] == "list":
    paths = glob.glob('model_configs/**/*.yml', recursive=True)
#show_menu([
#    "Train a model",
#    "Evaluate a model"
#])