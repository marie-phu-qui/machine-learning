// BUILD OUR SEQUENTIAL MODEL
const model = tf.sequential()
console.log(model)

// Config first layer = inputShape mandatory 
const configHid = {
  units: 3,
  inputShape: [2],
  activation: 'sigmoid'
}

const hidden = tf.layers.dense(configHid)
model.add(hidden)

// Config Second layer  
const configOut = {
  units: 1,
  activation: 'sigmoid'
}

const output = tf.layers.dense(configOut)
model.add(output)

// Optimizer with gradient descent
const sgdOpt = tf.train.sgd(0.1)

// Compile the code
const config = {
  optimizer: sgdOpt,
  loss: 'meanSquaredError'
}
model.compile(config)

// Training Data =input - has to be a number
const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1],
])
// Training Data =output- has to be a number
const ys = tf.tensor2d([
  [1],
  [1],
  [1],
])

const configFit = {
  shuffle: true,
  verbose: true,
  epochs: 10
}

async function train() {
  //the bigger the number in this loop size the lower the loss will be at the end (=more training)
  for (i = 0; i < 100; i++) {
    const response = await model.fit(xs, ys, configFit);
    console.log(response.history.loss[0])
  }
  // ONCE YOU ARE TRAINED PREDICT
  let outputs = model.predict(xs)
  outputs.print()
}
train().then(() => console.log('I did that'))



// THIS PROCESS A PREDICTION - NOT TRAINED - just random
// We need inputs from a tensor
// const xs = tf.tensor2d([
  //   [0.25, 0.92],
  //   [0.95, 0.24],
  //   [0.45, 0.23],
  //   [0.5, 0.91],
  //   [0.25, 0.92],
  // ])


  // let ys = model.predict(xs)

  // ys.print()