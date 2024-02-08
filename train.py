def train(data, net, epochs, opt, loss_func):
    loss_hist = []
    for epoch in range(epochs):
        for seq, label in data:
            label = label.reshape(len(label))
            opt.zero_grad()
            y_pre = net(seq)
            loss = loss_func(y_pre, label)
            loss.backward()
            opt.step()
        print("epoch ",epoch+1,": ",loss.item())
        loss_hist.append(loss.item())
    return net, loss_hist
