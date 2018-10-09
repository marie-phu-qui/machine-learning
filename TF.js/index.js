// BUILD OUR DATASET
function draw() {
  const values = [];
  for (let i = 0; i < 150000; i++) {
    values[i] = Math.floor(Math.random() * 100);
  }
  const shape = [500, 300];

  //   const test = tf.tensor2d(values, shape)
  // test.dispose()

  tf.tidy(() => {
    const a = tf.tensor2d(values, shape, 'int32')
    const b = tf.tensor2d(values, shape, 'int32')
    const b_t = b.transpose()
    const c = a.matMul(b_t)
    c.print();
  })
  // a.dispose();
  // b.dispose();
  // c.dispose();
  // b_t.dispose()
}
draw()

// BUILD OUR MODEL
const model = tf.sequential()

const configHid = {
  units: 4,
  activation: 'sigmoid'
}

const hidden = tf.layers.dense(configHid)

const configOut = {
  units: 3,
  activation: 'sigmoid'
}

const output = tf.layers.dense(configOut)

model.addLayer(hidden)
model.addLayer(output)