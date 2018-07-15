const tf = require('@tensorflow/tfjs');

// Crerate a Simple Model:
const model = tf.sequential();

// Create Hidden Layer
// Dense is a Fully Connected Layer
const hidden = tf.layers.dense({
  units: 100, // Number of nodes
  inputShape: [10],
  activation: 'sigmoid'
});

model.add(hidden);

const output = tf.layers.dense({
  units: 1,
  // Input layer inferred from previous layer
  activation: 'sigmoid'
});

model.add(output);

// Optimizer Using Gradient Descent
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]); // Input
const ys = tf.randomNormal([100, 1]); // Expected Result

model.fit(xs, ys, { //Training the Model
  epochs: 100,
  callbacks: {
    onEpochEnd: async (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`);
    }
  }
});
