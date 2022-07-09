from config import cfg
from lib import init, Data, MoveNet, Task
from lib.data.data_reader import data_read2memory


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

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    # model.build(input_shape=(1, 192, 192, 3))

    dataset, datasetval = data_read2memory(cfg)
    data = Data(cfg, dataset, datasetval)
    train_loader, val_loader, train_len, val_len = data.getTrainValDataloader()

    run_task = Task(cfg, model)
    run_task.train(train_loader, val_loader, train_len, val_len)

    # print(model.trainable_variables)
    # print(model.summary())

    # output = model(tf.ones((1, 3, 192, 192)))
    #
    # print(model.variables)


if __name__ == '__main__':
    main(cfg)
