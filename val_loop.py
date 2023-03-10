from pathlib import Path

import torch


# tensorboard
def validation(model, val_data, loss_fn, aud_to_mel, out_dir, lr, epoch, writer, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    print('Validating...')
    with torch.no_grad():
        for batch in val_data:
            x, y = batch

            mel = aud_to_mel(x)
            out = model(mel.transpose(1, 2).to(device))
            loss = loss_fn(out, y.to(device))

            running_loss += loss.item()

            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y.to(device)).sum().item()

    test_loss = running_loss / len(val_data)
    accu = correct / total
    writer.add_scalar("Loss/validation", loss, epoch)
    writer.add_scalar("Accuracy/validation", accu, epoch)
    print(f'Val_loss = {test_loss}, val_acc = {accu}')
    Path(f'{out_dir}/val_loss.txt').open('a').write(f'Val_loss = {test_loss}, val_acc = {accu}, lr = {lr}\n')
