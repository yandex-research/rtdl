"Code from https://github.com/google-research/google-research/blob/master/tabnet/experiment_covertype.py"
# %%
from pathlib import Path

import numpy as np
import tensorflow as tf
import zero

import lib


# %%
def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class TabNet:
    """TabNet model class."""

    def __init__(
        self,
        num_features,
        columns,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        batch_momentum,
        virtual_batch_size,
        num_classes,
        epsilon=0.00001,
        **kwargs,
    ):
        """Initializes a TabNet instance.
        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.
        Returns:
          A TabNet instance.
        """

        self.columns = columns
        self.num_features = num_features
        self.num_classes = num_classes

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

    def encoder(self, data, reuse, is_training):
        """TabNet encoder model."""

        with tf.variable_scope("Encoder", reuse=reuse):

            # Reads and normalizes input features.
            # NOTE we do data normalization at a dataset level
            if self.columns:
                features = tf.feature_column.input_layer(data, self.columns)
            else:
                features = data

            # features = tf.layers.batch_normalization(
            #     features, training=is_training, momentum=self.batch_momentum
            # )
            batch_size = tf.shape(features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complemantary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features]
            )
            total_entropy = 0

            if is_training:
                v_b = self.virtual_batch_size
            else:
                v_b = 1

            for ni in range(self.num_decision_steps):

                # Feature transformer with two shared and two decision step dependent
                # blocks is used below.

                reuse_flag = ni > 0

                transform_f1 = tf.layers.dense(
                    masked_features,
                    self.feature_dim * 2,
                    name="Transform_f1",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f1 = tf.layers.batch_normalization(
                    transform_f1,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f1 = glu(transform_f1, self.feature_dim)

                transform_f2 = tf.layers.dense(
                    transform_f1,
                    self.feature_dim * 2,
                    name="Transform_f2",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f2 = tf.layers.batch_normalization(
                    transform_f2,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f2 = (
                    glu(transform_f2, self.feature_dim) + transform_f1
                ) * np.sqrt(0.5)

                transform_f3 = tf.layers.dense(
                    transform_f2,
                    self.feature_dim * 2,
                    name="Transform_f3" + str(ni),
                    use_bias=False,
                )
                transform_f3 = tf.layers.batch_normalization(
                    transform_f3,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f3 = (
                    glu(transform_f3, self.feature_dim) + transform_f2
                ) * np.sqrt(0.5)

                transform_f4 = tf.layers.dense(
                    transform_f3,
                    self.feature_dim * 2,
                    name="Transform_f4" + str(ni),
                    use_bias=False,
                )
                transform_f4 = tf.layers.batch_normalization(
                    transform_f4,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f4 = (
                    glu(transform_f4, self.feature_dim) + transform_f3
                ) * np.sqrt(0.5)

                if ni > 0:

                    decision_out = tf.nn.relu(transform_f4[:, : self.output_dim])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keep_dims=True) / (
                        self.num_decision_steps - 1
                    )
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = transform_f4[:, self.output_dim :]

                if ni < self.num_decision_steps - 1:

                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    mask_values = tf.layers.dense(
                        features_for_coef,
                        self.num_features,
                        name="Transform_coef" + str(ni),
                        use_bias=False,
                    )
                    mask_values = tf.layers.batch_normalization(
                        mask_values,
                        training=is_training,
                        momentum=self.batch_momentum,
                        virtual_batch_size=v_b,
                    )
                    mask_values *= complemantary_aggregated_mask_values
                    mask_values = tf.contrib.sparsemax.sparsemax(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complemantary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values
                    )

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(
                            -mask_values * tf.log(mask_values + self.epsilon), axis=1
                        )
                    ) / (self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, features)

                    # Visualization of the feature selection mask at decision step ni
                    tf.summary.image(
                        f"Mask_for_step_{ni}",
                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                        max_outputs=1,
                    )

            # Visualization of the aggregated feature importances
            tf.summary.image(
                "Aggregated_mask",
                tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
                max_outputs=1,
            )

            return output_aggregated, total_entropy

    def classify(self, activations, reuse):
        """TabNet classify block."""

        with tf.variable_scope("Classify", reuse=reuse):
            logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
            predictions = tf.nn.softmax(logits)
            return logits, predictions

    def regress(self, activations, reuse):
        """TabNet regress block."""

        with tf.variable_scope("Regress", reuse=reuse):
            predictions = tf.layers.dense(activations, 1)
            return predictions


def make_tf_loaders(args, X, Y):
    tf.set_random_seed(args["seed"])
    datasets = {k: tf.data.Dataset.from_tensor_slices((X[k], Y[k])) for k in X.keys()}
    datasets = {k: tf.data.Dataset.from_tensor_slices((X[k], Y[k])) for k in X.keys()}
    X_loader = {}
    Y_loader = {}

    for k in datasets.keys():
        if k == lib.TRAIN:
            datasets[k] = datasets[k].shuffle(
                buffer_size=50, reshuffle_each_iteration=True
            )
            datasets[k] = datasets[k].batch(
                args["training"]["batch_size"], drop_remainder=True
            )
        else:
            datasets[k] = datasets[k].batch(args["training"]["batch_size"])

        # NOTE +1 for the final validation step for the best model at the end
        datasets[k] = datasets[k].repeat(args["training"]["epochs"] + 1)
        datasets[k] = datasets[k].make_initializable_iterator()

        X_loader[k], Y_loader[k] = datasets[k].get_next()

    # Add train with no shuffle dataset for final eval
    ds = tf.data.Dataset.from_tensor_slices((X[lib.TRAIN], Y[lib.TRAIN]))
    ds = ds.batch(args["training"]["batch_size"])
    ds = ds.make_initializable_iterator()
    k = "train_noshuffle"
    datasets[k] = ds
    X_loader[k], Y_loader[k] = ds.get_next()

    return datasets, X_loader, Y_loader


# %%
def get_train_eval_ops(args, data: lib.Dataset, model, x, y):
    "Create train step, train loss, val/test predict ops"
    encoder_out_train, total_entropy = model.encoder(
        x[lib.TRAIN], reuse=False, is_training=True
    )
    encoder_out_val, _ = model.encoder(x[lib.VAL], reuse=True, is_training=False)
    encoder_out_test, _ = model.encoder(x[lib.TEST], reuse=True, is_training=False)
    encoder_out_train_noshuffle, _ = model.encoder(
        x["train_noshuffle"], reuse=True, is_training=False
    )
    train_op = None

    # Regression and classification losses
    if data.is_multiclass:
        y_pred_train, _ = model.classify(encoder_out_train, reuse=False)
        y_pred_train_noshuffle, _ = model.classify(
            encoder_out_train_noshuffle, reuse=True
        )
        y_pred_val, _ = model.classify(encoder_out_val, reuse=True)
        y_pred_test, _ = model.classify(encoder_out_test, reuse=True)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            y[lib.TRAIN], y_pred_train, reduction=tf.losses.Reduction.MEAN
        )
        train_loss_op = (
            cross_entropy + args["training"]["sparsity_loss_weight"] * total_entropy
        )
    if data.is_regression:
        y_pred_train = model.regress(encoder_out_train, reuse=False)
        y_pred_train_noshuffle = model.regress(encoder_out_train_noshuffle, reuse=True)
        y_pred_val = model.regress(encoder_out_val, reuse=True)
        y_pred_test = model.regress(encoder_out_test, reuse=True)
        mse = tf.losses.mean_squared_error(
            tf.expand_dims(y[lib.TRAIN], axis=1),
            y_pred_train,
            reduction=tf.losses.Reduction.MEAN,
        )
        train_loss_op = mse + args["training"]["sparsity_loss_weight"] * total_entropy
    if data.is_binclass:
        y_pred_train = model.regress(encoder_out_train, reuse=False)
        y_pred_train_noshuffle = model.regress(encoder_out_train_noshuffle, reuse=True)
        y_pred_val = model.regress(encoder_out_val, reuse=True)
        y_pred_test = model.regress(encoder_out_test, reuse=True)
        log_loss = tf.losses.log_loss(
            tf.expand_dims(y[lib.TRAIN], axis=1),
            tf.nn.sigmoid(y_pred_train),
            reduction=tf.losses.Reduction.MEAN,
        )
        train_loss_op = (
            log_loss + args["training"]["sparsity_loss_weight"] * total_entropy
        )

    # Optimization step
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(
        global_step=global_step,
        **args["training"]["schedule"],
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(train_loss_op)
        grad_thresh = args["training"]["grad_thresh"]
        capped_gvs = [
            (tf.clip_by_value(grad, -grad_thresh, grad_thresh), var)
            for grad, var in gvs
        ]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    y_pred_op = {
        lib.TRAIN: y_pred_train_noshuffle,
        lib.VAL: y_pred_val,
        lib.TEST: y_pred_test,
    }

    return train_op, train_loss_op, y_pred_op


def evaluate(args, sess, y, y_pred_op, parts, task_type, y_info):
    metrics = {}
    predictions = {}

    for part in parts:
        _, epoch_size = lib.get_epoch_parameters(
            y[part].shape[0], args["training"]["batch_size"]
        )
        y_pred = []
        loader = lib.IndexLoader(y[part].shape[0], batch_size, False, "cpu")
        loader = iter(loader)

        for _ in range(epoch_size):
            if use_placeholders:
                idx = next(loader).numpy()
                feed_dict = {
                    x_loader[part]: X[part][idx],
                    y_loader[part]: Y[part][idx],
                }
                feed_dict[x_loader["train_noshuffle"]] = X[lib.TRAIN][idx]
                feed_dict[y_loader["train_noshuffle"]] = Y[lib.TRAIN][idx]
            else:
                feed_dict = None

            y_pred.append(sess.run(y_pred_op[part], feed_dict=feed_dict))
        y_pred = np.concatenate(y_pred)
        metrics[part] = lib.calculate_metrics(
            task_type, y[part], y_pred, 'logits', y_info
        )
        predictions[part] = y_pred

    for part, part_metrics in metrics.items():
        print(f'[{part:<5}]', lib.make_summary(part_metrics))

    return metrics, predictions


# %%
args, output = lib.load_config()
zero.set_randomness(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats = {
    "dataset": args["data"]["path"],
    "algorithm": Path(__file__).stem,
    **lib.load_json(output / "stats.json"),
}

tf.reset_default_graph()
tf.set_random_seed(args["seed"])

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'counter'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
if not isinstance(X, tuple):
    X = (X, None)

zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')
X_num, X_cat = X

use_placeholders = D.info["name"] in ["epsilon", "yahoo"]
columns = None

if use_placeholders:
    X = X_num
    # Epsilon dataset doesn't work with tf data Dataset well
    x_loader = {
        k: tf.placeholder(tf.float32, shape=(None, D.info['n_num_features']))
        for k in X.keys()
    }
    y_loader = {k: tf.placeholder(tf.int32, shape=(None,)) for k in Y.keys()}

    x_loader["train_noshuffle"] = tf.placeholder(
        tf.float32, shape=(None, D.info['n_num_features'])
    )
    y_loader["train_noshuffle"] = tf.placeholder(tf.float32, shape=(None,))
else:
    if X_cat is not None:
        X = {}
        for part in lib.PARTS:
            X[part] = {}
            for i in range(X_num[part].shape[1]):
                X[part][str(i)] = X_num[part][:, i]
            for i in range(
                X_num[part].shape[1], X_num[part].shape[1] + X_cat[part].shape[1]
            ):
                X[part][str(i)] = X_cat[part][:, i - X_num[part].shape[1]]
    else:
        X = X_num

    datasets_tf, x_loader, y_loader = make_tf_loaders(args, X, Y)

    if X_cat is not None:
        num_columns = [
            tf.feature_column.numeric_column(str(i))
            for i in range(X_num['train'].shape[1])
        ]
        cat_columns = [
            tf.feature_column.categorical_column_with_identity(
                str(i), max(X_cat['train'][:, i - X_num['train'].shape[1]]) + 1
            )
            for i in range(
                X_num['train'].shape[1],
                X_num['train'].shape[1] + X_cat['train'].shape[1],
            )
        ]
        emb_columns = [
            tf.feature_column.embedding_column(c, args["model"]["d_embedding"])
            for c in cat_columns
        ]
        columns = num_columns + emb_columns

# Restricting hyperparameter search space from original paper
# 1. N_a = N_b

args["model"]["output_dim"] = args["model"]["feature_dim"]
print(columns)

model = TabNet(
    num_classes=D.info['n_classes'] if D.is_multiclass else 1,
    columns=columns,
    num_features=X_num['train'].shape[1] + 0
    if X_cat is None
    else args["model"]["d_embedding"] * X_cat['train'].shape[1],
    **args["model"],
)

train_op, train_loss_op, y_pred_op = get_train_eval_ops(
    args, D, model, x_loader, y_loader
)

init = tf.initialize_all_variables()
init_local = tf.local_variables_initializer()
init_table = tf.tables_initializer(name="Initialize_all_tables")

batch_size = stats['batch_size'] = args["training"]["batch_size"]
epoch_size = stats['epoch_size'] = (
    Y[lib.TRAIN].shape[0] // batch_size
)  # drop_last=True in tf
progress = zero.ProgressTracker(args["training"]["patience"])
saver = tf.train.Saver()

timer = zero.Timer()
timer.run()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_local)
    sess.run(init_table)
    if not use_placeholders:
        for k in datasets_tf.keys():
            sess.run(datasets_tf[k].initializer)

    for e in range(args["training"]["epochs"]):
        epoch_timer = zero.Timer()
        epoch_timer.run()

        loader = lib.IndexLoader(Y[lib.TRAIN].shape[0], batch_size, True, "cpu")
        loader = iter(loader)

        for step in range(epoch_size):
            if use_placeholders:
                idx = next(loader)

                feed_dict = {
                    x_loader[lib.TRAIN]: X[lib.TRAIN][idx],
                    y_loader[lib.TRAIN]: Y[lib.TRAIN][idx],
                }
            else:
                feed_dict = None

            if step % args["training"]["display_steps"] == 0:
                _, train_loss = sess.run([train_op, train_loss_op], feed_dict=feed_dict)
                print(f"Step {step}, Train Loss {train_loss:.4f}")
            else:
                _ = sess.run(train_op, feed_dict=feed_dict)

        print(f"Epoch {e} done; time {zero.format_seconds(epoch_timer())}")
        metrics, predictions = evaluate(
            args, sess, Y, y_pred_op, [lib.VAL, lib.TEST], D.info['task_type'], y_info
        )
        progress.update(metrics[lib.VAL]["score"])

        if progress.success:
            print("New best epoch")
            stats["best_epoch"] = e
            saver.save(sess, str(output / "checkpoint.ckpt"))
            lib.dump_stats(stats, output, final=False)
        elif progress.fail:
            print("Early stopping")
            break

    saver.restore(sess, str(output / "checkpoint.ckpt"))
    stats['metrics'], predictions = evaluate(
        args, sess, Y, y_pred_op, lib.PARTS, D.info['task_type'], y_info
    )

    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)

    saver.save(sess, str(output / "best.ckpt"))
    lib.dump_stats(stats, output, final=True)

print(f"Total time: {zero.format_seconds(timer())}")
