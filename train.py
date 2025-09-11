import kagglehub
import tensorflow as tf
import os
from model.densenet import DenseNet
from torch.utils.data import DataLoader, Dataset
from data.preprocessing import Preprocessor

def load_data(root, split, img_shape=(224,224), batch_size=32, augment=True):
    split_dir = os.path.join(root, "chest_xray", split)

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=img_shape,
        batch_size=batch_size
    )

    preprocessor = Preprocessor(img_shape=img_shape, augment=augment)
    
    def preprocessing_fn(img, label):
        return preprocessor(img, training=(split=='train')), label

    ds = ds.map(preprocessing_fn)

    if split == 'train':
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
    
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = DenseNet()
    root = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
    train_loader = load_data(root, split='train', augment=True)
    val_loader = load_data(root, split='test', augment=False)

    model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics = ['accuracy']

    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    model.fit(train_loader, validation_data=val_loader, epochs=100, callbacks=[checkpoint, lr_scheduler])
