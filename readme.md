# resume training example

A example of showing how to resume training your network from crash in middle.

blog post: https://dreamanddead.github.io/2020/03/04/resume-training-from-crash.html

## usage

### train

```sh
$ python train.py -o output
```

训练过程会将所有中间状态和训练产出保存在 `output/` 文件夹下。

训练过程随时可以用 `Ctrl-C` 中断，再次执行命令，可继续上次中断处训练。

### eval

```sh
$ python eval.py -m output -d data
```

`data/` 中是几张自己手写的图片。

`eval` 过程会从 `output/` 中加载模型和编码表，预测图片对应的数字。
