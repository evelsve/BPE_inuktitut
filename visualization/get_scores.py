import matplotlib.pyplot as plt


def load_file(file, valid=False):
    with open(file, 'r', encoding="utf8") as file:
        lines = file.readlines()
        list_loss = []
        list_epoch = []
        for line in lines:
            split = line.split()
            epoch = int(split[8].strip(":"))
            if valid==False:
                loss = float(split[12].split("=")[1].strip(","))
            else:
                loss = float(split[16])
            list_loss.append(loss)
            list_epoch.append(epoch)
    return list_loss, list_epoch

def load_bleu(valid_file):
    with open(valid_file, 'r', encoding="utf8") as file:
        lines = file.readlines()
        list_bleu = []
        list_epoch = []
        for line in lines:
            split = line.split()
            epoch = int(split[8].strip(":"))
            bleu = float(split[25])
            list_bleu.append(bleu)
            list_epoch.append(epoch)
    return list_bleu, list_epoch


train_loss, train_epoch = load_file("score_training.txt")
valid_loss, valid_epoch = load_file("score_validation.txt", valid=True)

plt.scatter(train_epoch, train_loss)
plt.scatter(valid_epoch, valid_loss)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training loss", "Validation loss"])
plt.figure()


bleu, epoch = load_bleu("score_validation.txt")
plt.plot(epoch, bleu)
plt.ylabel("BLEU")
plt.xlabel("Epoch")
plt.show()