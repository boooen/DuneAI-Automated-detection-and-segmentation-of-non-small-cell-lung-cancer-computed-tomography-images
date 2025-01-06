import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import numpy as np

# Define the Dice Similarity Coefficient (DSC) and Dice Loss
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Define the Combined Loss Function (DSC + Binary Cross-Entropy)
def combined_loss(y_true, y_pred, alpha=0.7):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return alpha * dsc + (1 - alpha) * bce

# Cosine Annealing Learning Rate Scheduler 学习率调度器
def cosine_annealing_scheduler(epoch, lr):
    initial_lr = 0.001  # Initial learning rate
    max_epochs = 300    # Total number of epochs
    min_lr = 0.0001     # Minimum learning rate
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    return min_lr + (initial_lr - min_lr) * cosine_decay

# Training Function
def train_model(model, train_generator, validation_generator, epochs=300, initial_lr=0.001, min_lr=0.0001):
    """
    Train the model using the combined loss function and cosine annealing learning rate scheduler.

    Args:
        model: The Keras model to be trained.
        train_generator: Data generator for training data.
        validation_generator: Data generator for validation data.
        epochs: Number of training epochs (default: 300).
        initial_lr: Initial learning rate (default: 0.001).
        min_lr: Minimum learning rate for cosine annealing (default: 0.0001).

    Returns:
        history: Training history object.
    """
    # Checkpoint callback to save the best model based on DSC
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',  # Save as .hdf5 format
        monitor='val_dice_coefficient',  # Monitor DSC on validation set
        save_best_only=True, # 设置为 True 表示只有当 monitor 指定的指标比之前更好时，才会保存模型权重。
        mode='max',  # Save the model with the highest DSC
        verbose=1,
        save_weights_only=True  # Save only the model weights
    )

    # Compile the model with the combined loss function
    # 虽然编译时使用Adam优化器，实际上由于回调函数中的处理，学习率调整器用的是 LearningRateScheduler(cosine_annealing_scheduler)
    model.compile(optimizer=Adam(learning_rate=initial_lr), loss=combined_loss, metrics=[dice_coefficient])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint, LearningRateScheduler(cosine_annealing_scheduler)]
    )

    # Save the final model weights (optional)
    model.save_weights('final_model_weights.h5')  # Save final weights as .hdf5

    return history