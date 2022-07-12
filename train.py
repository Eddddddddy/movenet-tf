from config import cfg
from lib import init, Data, MoveNet, Task
from lib.data.data_reader import data_read2memory
import tensorflow as tf
from lib.loss.movenet_loss_o import MovenetLoss
import os

import datetime


def main(cfg):
    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/tmp/tfdbg2_logdir",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    init(cfg)
    # model = Backbone()
    # model.build(input_shape=(1, 192, 192, 3))
    # print(model.summary())
    #
    # model = Header(cfg["num_classes"], mode='train')
    # model.build(input_shape=(1, 48, 48, 24))
    # # print(model.trainable_variables)
    # print(model.summary())

    # ======================================================================================




    print("===========================")
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')



    # ======================================================================================

    # model.build(input_shape=(1, 192, 192, 3))
    # print(model.summary())

    # dataset, datasetval = data_read2memory(cfg)
    data = Data(cfg)
    train_loader, val_loader, train_loader_x, train_loader_y, train_loader_z, train_len, val_len = data.getTrainValDataloader()

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'], clipvalue=cfg['clip_gradient']),
        loss=MovenetLoss(cfg=cfg))

    def generator():
        for input, output1, output2, output3 in train_loader:
            yield input, [output1, output2, output3]

    model.fit(train_loader_z,
              epochs=cfg["epochs"],
              verbose=2,
              batch_size=cfg["batch_size"],
              callbacks=[cp_callback])

    # run_task = Task(cfg, model)
    # run_task.train(train_loader, val_loader, train_len, val_len)

    # print(model.trainable_variables)
    # print(model.summary())

    # output = model(tf.ones((1, 3, 192, 192)))
    #
    # print(model.variables)


if __name__ == '__main__':
    main(cfg)
