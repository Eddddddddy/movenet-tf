from config import cfg
from lib import init, Data, MoveNet, Task
from lib.data.data_reader import data_read2memory
import tensorflow as tf
from lib.loss.movenet_loss_o import MovenetLoss


def main(cfg):
    init(cfg)
    # model = Backbone()
    # model.build(input_shape=(1, 192, 192, 3))
    # print(model.summary())
    #
    # model = Header(cfg["num_classes"], mode='train')
    # model.build(input_shape=(1, 48, 48, 24))
    # # print(model.trainable_variables)
    # print(model.summary())
    print("===========================")
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    # model.build(input_shape=(1, 192, 192, 3))
    # print(model.summary())

    # dataset, datasetval = data_read2memory(cfg)
    data = Data(cfg)
    train_loader, val_loader, train_loader_x, train_loader_y, train_len, val_len = data.getTrainValDataloader()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'], clipvalue=cfg['clip_gradient']),
        loss=MovenetLoss())

    def generator():
        for input, output1, output2, output3 in train_loader:
            yield input, [output1, output2, output3]

    model.fit(generator(), epochs=cfg["epochs"], verbose=2)

    # run_task = Task(cfg, model)
    # run_task.train(train_loader, val_loader, train_len, val_len)

    # print(model.trainable_variables)
    # print(model.summary())

    # output = model(tf.ones((1, 3, 192, 192)))
    #
    # print(model.variables)


if __name__ == '__main__':
    main(cfg)
