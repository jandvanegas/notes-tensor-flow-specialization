# Introduction to Computer Vision

The data used for this unit will be `Fashion MNIST` which has
* 70k images
* 10 categories
* Images are 28x28
* can train a neural net

## Loading the Data
```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
Using numeric labels is a first step to avoid human bias. [Here more info](https://developers.google.com/machine-learning/fairness-overview/)

## Writing the code
Now we'll have three layers
```
model = keras.Sequentioal([
	keras.layers.Flatten(input_shape(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmas)
	])
```
To deep more in the math concept, [here is a resource](https://youtu.be/fXOsFF95ifk)

A Jupyter notebook can be found [here](https://github.com/jandvanegas/dlaicourse/blob/393039e05c0772e6d70add45212d9e1b3c2686b9/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)

## Callbacks
In order to stop a training when it reaches some parameter you can use callbacks. Here is an example
```
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss') < 0.4):
			print("\nLoss is low so cancelling training")
			self.model.stop_training = True

callbacks = myCallback()

model.fit(training_data, training_labels, epoch=5, callbacks=[callback])
```
[Here](https://github.com/jandvanegas/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb) is the completed code.
