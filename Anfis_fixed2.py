import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

N = 300
Temp = 10 + 25 * np.random.rand(N, 1)         
TDS  = 50 + 1450 * np.random.rand(N, 1)      
NTU  = 0.01 * TDS.flatten() + 0.02 * (Temp.flatten() - 20) ** 2 + 5 * np.random.randn(N)

X = np.hstack([Temp, TDS])
y = NTU

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

class ANFIS(tf.keras.Model):
    def __init__(self, n_inputs=2, n_mfs=[3, 3]):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.centers = []
        self.sigmas = []
        for i in range(n_inputs):
            m = n_mfs[i]
            c_init = np.linspace(0.0, 1.0, m).astype(np.float32)
            s_init = (0.1 * np.ones(m)).astype(np.float32)
            c = self.add_weight(
                name=f"center_{i}",
                shape=(m,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(c_init),
                trainable=True,
            )
            s = self.add_weight(
                name=f"sigma_{i}",
                shape=(m,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(s_init),
                trainable=True,
            )
            self.centers.append(c)
            self.sigmas.append(s)
        self.num_rules = int(np.prod(n_mfs))
        pq_init = tf.random.normal([self.num_rules, n_inputs], stddev=0.1)
        r_init = tf.random.normal([self.num_rules], stddev=0.1)
        self.pq = self.add_weight(
            name="pq",
            shape=(self.num_rules, n_inputs),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(pq_init.numpy()),
            trainable=True,
        )
        self.r = self.add_weight(
            name="r",
            shape=(self.num_rules,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(r_init.numpy()),
            trainable=True,
        )
        grids = np.meshgrid(*[range(m) for m in n_mfs], indexing="ij")
        indices = np.stack([g.flatten() for g in grids], axis=1)
        self.rule_indices = tf.constant(indices, dtype=tf.int32)

    def gaussian(self, x, c, s):
        x = tf.expand_dims(x, axis=1)  
        return tf.exp(-0.5 * ((x - c) / s) ** 2)
    def call(self, X):
        batch = tf.shape(X)[0]
        mf_vals = []
        for i in range(self.n_inputs):
            c = self.centers[i]  
            s = tf.nn.softplus(self.sigmas[i]) + 1e-9
            xi = X[:, i]
            vals = self.gaussian(xi, c, s)  
            mf_vals.append(vals)
        res = mf_vals[0]
        for j in range(1, self.n_inputs):
            res = tf.einsum("bm,bn->bmn", res, mf_vals[j])
            res = tf.reshape(res, (batch, -1))
        firing = res  
        denom = tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-12
        w_bar = firing / denom  
        y_rules = tf.matmul(X, tf.transpose(self.pq)) + self.r  
        out = tf.reduce_sum(w_bar * y_rules, axis=1)
        return out

X_min = X_train.min(axis=0).astype(np.float32)
X_max = X_train.max(axis=0).astype(np.float32)

def scale(X):
    return (X - X_min) / (X_max - X_min + 1e-9)


Xtr_s = scale(X_train).astype(np.float32)
Xte_s = scale(X_test).astype(np.float32)
ytr = y_train.astype(np.float32)
yte = y_test.astype(np.float32)


model = ANFIS(n_inputs=2, n_mfs=[3, 3])

print("Jumlah trainable variables:", len(model.trainable_variables))
for v in model.trainable_variables:
    print(" -", v.name, v.shape)

epochs = 200
batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
mse_loss = tf.keras.losses.MeanSquaredError()

train_dataset = tf.data.Dataset.from_tensor_slices((Xtr_s, ytr)).shuffle(200).batch(batch_size)

loss_history = []
for ep in range(epochs):
    ep_losses = []
    for xb, yb in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(xb)
            loss = mse_loss(yb, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        if len(grads_and_vars) == 0:
            raise RuntimeError("No gradients available to apply. Check model connectivity to loss.")
        optimizer.apply_gradients(grads_and_vars)
        ep_losses.append(float(loss))
    mean_loss = np.mean(ep_losses)
    loss_history.append(mean_loss)
    if (ep + 1) % 20 == 0 or ep == 0:
        print(f"Epoch {ep+1}/{epochs}, train MSE = {mean_loss:.6f}")

pq_np = model.pq.numpy()  
r_np = model.r.numpy()  
ranges = (X_max - X_min).astype(np.float32)  
mins = X_min.astype(np.float32)


pq_orig = pq_np / ranges[np.newaxis, :]  
r_orig = r_np - np.sum(pq_np * (mins / ranges)[np.newaxis, :], axis=1)

print("\nAturan dalam satuan input ASLI:")
for i, idx in enumerate(model.rule_indices.numpy()):
    p0 = pq_orig[i, 0]
    p1 = pq_orig[i, 1]
    rr = r_orig[i]
    print(
        f"Rule {i+1}: IF Temp IS MF{idx[0]} AND TDS IS MF{idx[1]} "
        f"THEN NTU = {p0:.6f}*Temp + {p1:.6f}*TDS + {rr:.6f}"
    )

def plot_input_mfs(model, input_index, X_min, X_max, n_pts=200):
    x_orig = np.linspace(X_min[input_index], X_max[input_index], n_pts)
    x_scaled = (x_orig - X_min[input_index]) / (X_max[input_index] - X_min[input_index] + 1e-9)
    centers = model.centers[input_index].numpy()  
    sigmas = tf.nn.softplus(model.sigmas[input_index]).numpy()  
    plt.figure()
    for j, (c, s) in enumerate(zip(centers, sigmas)):
        mf = np.exp(-0.5 * ((x_scaled - c) / s) ** 2)
        plt.plot(x_orig, mf, label=f"MF{j}")
    plt.title(f"Membership functions for input {input_index} (0=Temp,1=TDS)")
    plt.xlabel("Original input value")
    plt.ylabel("Membership degree")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_input_mfs(model, 0, X_min, X_max)
plot_input_mfs(model, 1, X_min, X_max)

y_pred_tr = model(Xtr_s).numpy()
y_pred_te = model(Xte_s).numpy()

rmse_tr = np.sqrt(mean_squared_error(ytr, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(yte, y_pred_te))
mae_te = mean_absolute_error(yte, y_pred_te)
r2_te = r2_score(yte, y_pred_te)

print("\nEvaluasi:")
print(f"RMSE train = {rmse_tr:.4f}")
print(f"RMSE test  = {rmse_te:.4f}")
print(f"MAE  test  = {mae_te:.4f}")
print(f"R2   test  = {r2_te:.4f}")

plt.figure()
plt.plot(loss_history, "-b")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training loss")
plt.grid(True)

plt.figure()
plt.scatter(yte, y_pred_te, c="blue", edgecolor="k")
rng = [min(yte.min(), y_pred_te.min()), max(yte.max(), y_pred_te.max())]
plt.plot(rng, rng, "r--")
plt.xlabel("Actual NTU")
plt.ylabel("Predicted NTU")
plt.title(f"Predicted vs Actual (RMSE={rmse_te:.3f}, R2={r2_te:.3f})")
plt.grid(True)

plt.figure()
res = yte - y_pred_te
plt.subplot(2, 1, 1)
plt.hist(res, bins=20)
plt.title("Histogram Residuals")
plt.subplot(2, 1, 2)
plt.scatter(y_pred_te, res)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual vs Predicted")

plt.tight_layout()
plt.show()

print("\nDaftar aturan (kombinasi MF index untuk tiap input):")
for i, idx in enumerate(model.rule_indices.numpy()):
    print(f"Rule {i+1}: IF Temp IS MF{idx[0]} AND TDS IS MF{idx[1]} THEN NTU = linear(p,q,r)")
print("\nContoh parameter consequent (p,q,r) untuk beberapa rule (SKALA INTERNAL 0..1 inputs):")
for i in range(min(6, model.num_rules)):

    print(f"Rule {i+1} pq={pq_np[i]}, r={r_np[i]}")
