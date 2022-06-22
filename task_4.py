import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import PIL

# This method does not function completely correct, even if we try to generate millions of samples, we did not observe
# exceed of the critical value in the region that is defined in the exercise sheet. We think this is related to the
# issue that our model is not able to generate data very accurately.
def check_critical_region(region, train_fireEvac, test_fireEvac, model, epoch, latent_dim, display):
    """
    This method starts at a value and checks whether the critical value is exceeded in the given region.
    @param region: The region in which the number of pedestrians should not go higher than critical value of 100.
    @param train_fireEvac: training set
    @param test_fireEvac: test set
    @param model: nn that is used
    @param epoch: current epoch
    @param latent_dim: dimension of latent space
    @param display: display on which we plot
    @return: number of pedestrians, at which the generation concludes exceed of the critical value in the given region
    """
    threshold = 1000000000
    num_samples = 10000000

    # normalize region boundaries
    concat = np.concatenate((train_fireEvac, test_fireEvac))
    concat = np.concatenate((concat, region))
    concat = preprocess_data(concat, -1, 1)
    region = np.reshape(concat[-2:], (4,))

    # run until we exceed the critical number of pedestrian in the critical region
    while num_samples < threshold:
        print("Generating", num_samples, "samples:")
        # generate data
        generated_data = generate_and_save_images(model, epoch, num_samples, latent_dim)
        # check if critical area exceeds 100 people
        exceeds = False

        exceeded_people = 0
        # check whether sample is within the rectangle
        for sample in generated_data:
            if region[0] <= sample[0] <= region[1] and region[2] <= sample[1] <= region[3]:
                exceeded_people += 1

        # if number of people in rectangle over 100, we have found the limit
        exceeds |= exceeded_people > 100

        # find the number of pedestrians where the generation sample exceeds critical value
        if exceeds:
            print("exceeded at" + str(num_samples) + "ped generation")
            return num_samples

        display.clear_output(wait=False)
        num_samples += 1


def preprocess_data(data_set, target_min, target_max):
    """
    Preprocess the data to normalize into [min,max]
    :param data_set: the data set to normalize. Should be an numpy array of data of arbitrary dimension.
    :param target_min: min
    :param target_max: max
    """
    data_max = np.max(data_set)
    data_min = np.min(data_set)

    data_set = data_set - data_min  # normalize to (0, data_max - data_min)
    data_set /= (data_max - data_min)  # normalize to (0,1)
    data_set *= (target_max - target_min)  # normalize (0, target_max - target_min)
    data_set += target_min  # to (target_min, target_max)
    data_set = data_set.astype("float32")
    return data_set


class CVAE(tf.keras.Model):
    """
    Defines our variational autoencoder model. We use the following NNs: 
    - encoder [(2,)]: -> Dense[(2,)] = 512 -> Dense[(512)] = 512 -> Dense[(512)] = 512 ->   
                         Dense[(latent_dim + latent_dim)] 
    - decoder [(latent_dim)]: Dense[(latent_dim)] = 512 -> Dense[(512)] = 512 -> Dense[(512)] = 512 -> Dense[(512)] = 2
                              
    For the approximate posterior (latent space) we assume diagonal Gaussian.
    For the likelihood we assume (diagonal) unit Gaussian.
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(2,), name="encoder_input"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                      name="encoder_fully_connected_hidden_layer_1"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                      name="encoder_fully_connected_hidden_layer_2"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.tanh,
                                      name="encoder_fully_connected_hidden_layer_3"),
                # BI DE SUNU DENEYEBILIR MIYIM
                tf.keras.layers.Dense(units=latent_dim + latent_dim, activation=None, name="encoder_output"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,), name="decoder_input"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                      name="decoder_fully_connected_hidden_layer_1"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                      name="decoder_fully_connected_hidden_layer_2"),
                tf.keras.layers.Dense(units=512, activation=tf.keras.activations.tanh,
                                      name="decoder_fully_connected_hidden_layer_3"),
                tf.keras.layers.Dense(units=2, activation=None, name="decoder_output"),
                tf.keras.layers.Reshape(target_shape=(2,)),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        """
        samples 
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        # return self.decode(eps, apply_sigmoid=True)
        return self.decode(eps)

    def encode(self, x):
        """
        Encode the input tensor x using the Encoder defined in the VAE.
        :param x: Input tensor of shape (28,28)
        :returns: (mean, log(variance)) Parameters of the approximate posterior distribution. 
        """
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Computes z using the reparametrisation trick with the 
        multivariate diagonal Gaussian parameters. 
        :param mean: Mean of the approximate posterior 
                    (multivariate diagonal Gaussian is assumed)
        :param logvar: Logarithm of the variance of the approximate posterior 
                      (multivariate diagonal Gaussian is assumed)
        :returns: z = mean + \sqrt(var) * epsilon
                  epsilon is sampled from N(0,I)
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """
        Decode the latent sample z using the Decoder defined in the VAE.
        :param z: The sample tensor z in the latent space 
        :param apply_sigmoid: An option to apply sigmoid to the output of the decoder.
                              This was for sampling in the normalized space, but not used.
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    pdf of the logarithm of the Gaussian distribution
    :param sample: sample x to feed in the pdf 
    :param mean: mean of the normal distribution
    :param logvar: log(variance) of the normal distribution
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):  # lan burda ayni mi oluyor acaba loss
    """
    log(p(x)) >= ELBO = MC estimate of log(p(x|z)) + log(p(z)) - log(q(z|x))
    where we can compute the log probabilities of the distributions that we have assumed. 
    We assume diagonal normal with Identity covariance matrix for p(x|z)
    We assume diagonal normal for q(z|x)
    We assume diagonal normal with zero mean and identity covar matrix for the prior p(z)
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    reconst_loss = -tf.reduce_mean(0.5 * ((x - x_logit) ** 2))
    x_re = tf.reshape(x, shape=(x.get_shape()[0], -1))
    x_logit_re = tf.reshape(x_logit, shape=(x_logit.get_shape()[0], -1))
    reconst_likelihood = -tf.reduce_mean(tf.square(x_re - x_logit_re)) * 1000

    kl_div = tf.reduce_mean(logqz_x - logpz)

    elbo = reconst_likelihood - kl_div  # HELP MEE, ya dur yaa :*( aha bak olcak.. az kaldi
    loss = -elbo

    # GENERATION ERROR COK FAZLA AMINA KOYACAM BISI DENICEM

    return loss


@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and returns the loss. 
    Computes the loss and gradients.
    Updates the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def reconstruct_and_save_images(model, epoch, test_sample, image_name_prefix=""):
    """
    Reconstruct the given test images, plot and save them.
    :param model: Variational autoencoder
    :param epoch: current iteration number of the epoch
    :param test_sample: test images to reconstruct (THE WHOLE TEST SET)
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.decode(z)
    plt.figure()
    plt.scatter(predictions[:, 0], predictions[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./figures/TASK-4/TASK-4-reconstructions_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def sample_z(latent_dim, num_images_to_generate):
    """
    Samples z from the prior distribution standard normal. 
    :param latent_dim: number of dimensions of z (latent dimension)
    :param num_images_to_generate: number of total z to sample
    :return: sampled z's as numpy array
    """
    # latent variable list for each image
    samples = []
    for image in range(num_images_to_generate):
        for dim in range(latent_dim):
            # sample each latent dimension seperately (iid assumption)
            samples.append(tf.random.normal(shape=(1,)))

    return np.reshape(np.array(samples), (num_images_to_generate, latent_dim))


def generate_and_save_images(model, epoch, num_images_to_generate, latent_dim, image_name_prefix=""):
    """
    Generate randomly sampled images (sampled from the prior z distribution and then decoded)
    :param model: Variational Autoencoder
    :param epoch: current epoch 
    :param num_images_to_generate: number of images to generate
    :param latent_dim: number of latent dimensions == shape of z
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    # For this subtask we will first sample random latent variables (for 16 images)    
    z = sample_z(latent_dim, 1000)  # generate 1000 latent vars

    # generate predictions 
    predictions = model.decode(z)

    plt.figure()
    plt.scatter(predictions[:, 0], predictions[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('./figures/TASK-4/TASK-4-generations_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    return predictions


def plot_and_save_loss_curves(history_dict, epoch, image_name_prefix=""):
    """
    Plots and saves loss curves of the train and test datasets
    :param history_dict: dictionary containing the losses in each epoch for train and test
                         train loss should be in history_dict["train_loss"]
                         test loss should be in history_dict["test_loss"]
    :param epoch: total number of epochs trained with this network
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    plt.title('Loss curves')
    plt.plot(history_dict["train_loss"], '-', label='train')
    plt.plot(history_dict["test_loss"], '-', label='test')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel(r'$-\mathcal{L}_{ELBO}$')
    plt.savefig('./figures/TASK-4/TASK-4_loss_curves_till_epoch_{:04d}.png'.format(epoch))
    plt.show()

    # ZAMAN YETMIYOR
