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

def load_ppl(valid_file):
    with open(valid_file, 'r', encoding="utf8") as file:
        lines = file.readlines()
        list_ppl = []
        list_epoch = []
        for line in lines:
            split = line.split()
            # print(split)
            epoch = int(split[8].strip(":"))
            # print(epoch)
            ppl = float(split[22])
            list_ppl.append(ppl)
            # print(bleu)
            list_epoch.append(epoch)
    return list_ppl, list_epoch


train_loss, train_epoch = load_file("score_training.txt")
valid_loss, valid_epoch = load_file("score_validation.txt", valid=True)

plt.scatter(train_epoch, train_loss)
plt.scatter(valid_epoch, valid_loss)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training loss", "Validation loss"])
plt.figure()


bleu, epoch = load_ppl("score_validation.txt")
plt.plot(epoch, bleu, color='green')
plt.ylabel("Perplexity")
plt.xlabel("Epoch")
plt.show()