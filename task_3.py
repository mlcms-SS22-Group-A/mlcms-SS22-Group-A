import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import PIL


def preprocess_images(images):
    """
    :param images: one channel 28x28 images from the original mnist dataset
    :returns: normalized image with pixel values between 0 and 1
    """
    # convert the uint8 types to float32 
    images = images.astype("float32") 
    
    # 8uint type has the range [0,255] therefore it suffices to 
    # divide all the pixel values by 255 to normalize them between 0 and 1
    images = images / 255.0
    return images

class CVAE(tf.keras.Model):
    """
    Defines our variational autoencoder model. We use the following NNs:
    - encoder [(28,28)]: Flatten[(28,28)] = 28*28 = 784 -> Dense[(784)] = 256 -> Dense[(256)] = 256 ->   
                         Dense[(latent_dim + latent_dim)] 
    - decoder [(latent_dim)]: Dense[(latent_dim)] = 256 -> Dense[(256)] = 256 -> Dense[(256)] = 784 ->
                              Reshape[(784)] = (28,28)
                              
    For the approximate posterior (latent space) we assume diagonal Gaussian.
    For the likelihood we assume (diagonal) unit Gaussian.
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28), name="encoder_input"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu, name="encoder_fully_connected_hidden_layer_1"),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu, name="encoder_fully_connected_hidden_layer_2"),
                # No activation
                tf.keras.layers.Dense(units=latent_dim+latent_dim, activation=None, name="encoder_output"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,), name="decoder_input"),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu, name="decoder_fully_connected_hidden_layer_1"),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu, name="decoder_fully_connected_hidden_layer_2"),
                tf.keras.layers.Dense(units=784, activation=None, name="decoder_output"),
                tf.keras.layers.Reshape(target_shape=(28, 28)),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        """
        samples 
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        #return self.decode(eps, apply_sigmoid=True)
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
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    """
    log(p(x)) >= ELBO = MC estimate of log(p(x|z)) + log(p(z)) - log(q(z|x))
    where we can compute the log probabilities of the distributions that we have assumed. 
    We assume diagonal normal with Identity covariance matrix for p(x|z)
    We assume diagonal normal for q(z|x)
    We assume diagonal normal with zero mean and identity covar matrix for the prior p(z)
    """
    mean, logvar = model.encode(x)
    
    #plt.imshow(x.numpy()[1])
    
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    #plt.imshow(x_logit.numpy()[1])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    
    # MC estimate for the KL-divergence of q(x|z) to p(z)
    #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    #MC estimate
    #logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    #bce_loss = tf.reduce_mean(logpx_z)
    
    reconst_loss = -tf.reduce_mean(0.5 * ((x - x_logit) ** 2))
    x_re = tf.reshape(x, shape=(x.get_shape()[0], -1))
    x_logit_re = tf.reshape(x_logit, shape=(x_logit.get_shape()[0], -1))
    reconst_likelihood = -tf.reduce_mean(tf.square(x_re - x_logit_re)) * 1000

    #print("shape x:", x.get_shape())
    #print("shape x_logit:", x_logit.get_shape())
    #print("shape x_re:", x_re.get_shape())
    #print("shape x_logit_re:", x_logit_re.get_shape())
        
    kl_div = tf.reduce_mean(logqz_x - logpz)
    
    elbo = reconst_likelihood - kl_div 
    #elbo = -bce_loss - kl_div
    loss = -elbo
    
    #raise ValueError("WAIT")
    return loss

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    #MC estimate
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # Remark: logvariance is 0, the variance is identity for the prior
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


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


def split_mnist(images, labels):
    """
    Returns a dictionary containing each image according to their labels. 
    e.g.  mnist_dict["0"] contains only the images of the digit '0'.
    :param images: numpy array of the mnist images
    :param labels: labels of the corresponding image array
    :return: Dictionary of the split mnist dataset according to the labels
    """
    mnist_dict = {}
    mnist_dict["0"] = []
    mnist_dict["1"] = []
    mnist_dict["2"] = []
    mnist_dict["3"] = []
    mnist_dict["4"] = []
    mnist_dict["5"] = []
    mnist_dict["6"] = []
    mnist_dict["7"] = []
    mnist_dict["8"] = []
    mnist_dict["9"] = []

    index = 0 
    for label in labels:
        label_str = str(label)
        mnist_dict[label_str].append(images[index])
        index += 1
    # return the dictionary 
    return mnist_dict


def compute_plot_and_save_latent_representations(model, epoch, images, labels, image_name_prefix=""):
    """
    Runs the encoder and generates a 2d representation for the test dataset
    using the Encoder of the given model.
    :param model: Variational Autoencoder
    :param epoch: total number of epochs iterated in the training
    :param images: images of the dataset to compute the latent representations
    :param labels: labels of the corresponding images
    :param image_name_prefix: prefix of the saved image name (optional)
    """    
    # compute the latent representation with the initial model weights on the test set
    z_mean, z_logvar = model.encode(images)
    z = model.reparameterize(z_mean, z_logvar)
    image_dict = split_mnist(z, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    # scatter plot for all the digit with different colors
    ax.scatter(np.array(image_dict["0"])[:, 0], np.array(image_dict["0"])[:, 1], color="#006400")  # darkgreen
    ax.scatter(np.array(image_dict["1"])[:, 0], np.array(image_dict["1"])[:, 1], color="#00008b")  # darkblue    
    ax.scatter(np.array(image_dict["2"])[:, 0], np.array(image_dict["2"])[:, 1], color="#b03060")  # marron3 ~ pink
    ax.scatter(np.array(image_dict["3"])[:, 0], np.array(image_dict["3"])[:, 1], color="#ff0000")  # red
    ax.scatter(np.array(image_dict["4"])[:, 0], np.array(image_dict["4"])[:, 1], color="#ffff00")  # yellow
    ax.scatter(np.array(image_dict["5"])[:, 0], np.array(image_dict["5"])[:, 1], color="#00ff00")  # lime
    ax.scatter(np.array(image_dict["6"])[:, 0], np.array(image_dict["6"])[:, 1], color="#00ffff")  # aqua 
    ax.scatter(np.array(image_dict["7"])[:, 0], np.array(image_dict["7"])[:, 1], color="#ff00ff")  # purple
    ax.scatter(np.array(image_dict["8"])[:, 0], np.array(image_dict["8"])[:, 1], color="#6495ed")  # blue
    ax.scatter(np.array(image_dict["9"])[:, 0], np.array(image_dict["9"])[:, 1], color="#ffdead")  # sand ~ brown     
    
    ax.set_xlabel("latent dimension 1")
    ax.set_ylabel("latent dimension 2")
    ax.set_title("Latent Space Visualization after {:04d} full iteration (epoch)".format(epoch))
    fig.savefig("./figures/" + image_name_prefix + "latent_space_epoch_{:04d}.png".format(epoch))
    

def reconstruct_and_save_images(model, epoch, test_sample, image_name_prefix=""):
    """
    Reconstruct the given test images, plot and save them.
    :param model: Variational autoencoder
    :param epoch: current iteration number of the epoch
    :param test_sample: test images to reconstruct
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.decode(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig("./figures/" + image_name_prefix + 'reconstructions_at_epoch_{:04d}.png'.format(epoch))
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
    z = sample_z(latent_dim, 16)

    # generate predictions 
    predictions = model.decode(z)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig("./figures/" + image_name_prefix + 'randomly_generated_image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    
    
def plot_latent_images(model, epoch, num_images_to_generate, image_name_prefix=""):
    """
    Generates and plots images using the Decoder of the network. Latent space z is sampled with linearly evened spaces. 
    :param model: Variational Autoencoder
    :param epoch: The current epoch iteration
    :param num_images_to_generate: Number of images to generate
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    n = num_images_to_generate
    digit_size = 28
    norm = tfp.distributions.Normal(0, 1)

    # sample evenly spaced latent spaces 1 and 2 
    grid_latent_z_1 = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_latent_z_2 = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_latent_z_1):
        for j, xi in enumerate(grid_latent_z_2):
            z = np.array([[xi, yi]])
            x_decoded = model.decode(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size, 
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.savefig("./figures/" + image_name_prefix + 'latent_images_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

    
def display_image(epoch_no):
    """
    Displays the saved images (for the reconstructions)
    :param epoch_no: which epoch iteration to display 
    """
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def plot_and_save_loss_curves(history_dict, epoch, image_name_prefix=""):
    """
    Plots and saves loss curves of the train and test datasets
    :param history_dict: dictionary containing the losses in each epoch for train and test
                         train loss should be in history_dict["train_loss"]
                         test loss should be in history_dict["test_loss"]
    :param epoch: total number of epochs trained with this network
    :param image_name_prefix: prefix of the saved image name (optional)
    """
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.title('Loss curves')
    plt.plot(history_dict["train_loss"], '-', label='train')
    plt.plot(history_dict["test_loss"], '-', label='test')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel(r'$-\mathcal{L}_{ELBO}$')
    plt.show()
    plt.savefig("./figures/" + image_name_prefix + 'loss_curves_till_epoch_{:04d}.png'.format(epoch))
