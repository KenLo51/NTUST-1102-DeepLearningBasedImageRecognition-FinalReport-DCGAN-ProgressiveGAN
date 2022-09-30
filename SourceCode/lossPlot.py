from matplotlib import pyplot as plt

losses = {"generator":[], "discriminator":[]}

with open("DCGAN/loss_log.txt", mode="r", encoding="utf8") as lossFile:
    fileData = lossFile.read()

fileData = fileData.split("\n")
for lossStr in fileData[1:-1]:
    losses["generator"].append(float(lossStr[7:22]))
    losses["discriminator"].append(float(lossStr[32:]))

plt.plot(losses["generator"])
plt.plot(losses["discriminator"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(['generator','discriminator'])
plt.show()