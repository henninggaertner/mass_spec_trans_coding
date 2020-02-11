"""Components for encoding."""
import logging
import xarray as xr
import tensorflow as tf
import tensorflow_hub as hub
from .core import Component


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s;%(levelname)s;%(message)s',
    '%H:%M:%S'
))
logger.addHandler(stream_handler)


class Encoder(Component):
    def __init__(self, attributes={}):
        """
        Initialize the encoder.

        Args:
            attributes (dict): attributes to add to the resulting xr.DataArray.
        """
        self.attributes = attributes

    def __call__(self, an_object):
        """
        Encoding samples from an object and return results in xr.DataArray.

        Args:
            an_object (object): an object containing the data to be encoded.

        Returns:
            an object.
        """
        raise NotImplementedError


class HubEncoder(Encoder):
    def __init__(self, hub_module, batch_size=32, **kwargs):
        self.batch_size = batch_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            logger.debug("load module")
            self.module = hub.Module(hub_module)
            logger.debug("load module done")
            self.height, self.width = hub.get_expected_image_size(self.module)
            kwargs.update({'encoded_image_size': (self.height, self.width)})
            super(HubEncoder, self).__init__(attributes=kwargs)

            logger.debug("tf.data pipeline definition")

            def _resize_images(dataset, height=self.height, width=self.width):
                dataset = tf.cond(
                    tf.equal(tf.shape(dataset)[-1], 3), lambda: dataset,
                    lambda: tf.image.grayscale_to_rgb(dataset))
                dataset = tf.image.resize_images(dataset, (height, width))
                return dataset

            self.data = tf.placeholder(tf.float32,
                                       shape=[None, None, None, None])
            dataset = tf.data.Dataset.from_tensor_slices(self.data)
            dataset = dataset.map(_resize_images)
            dataset = dataset.batch(batch_size)  # single batch
            self.iterator = dataset.make_initializable_iterator()
            next_batch = self.iterator.get_next()
            self.features = self.module(next_batch)

    def __call__(self, data_array):
        """
        Encoding images with a tensorflow hub module.
        The images are resized to fit the module.

        Args:
            data_array (xarray.DataArray): expected dims are sample,
                height, width, channel. Length of channel must be 1 or 3.

        Returns:
            a xr.DataArray.
        """

        zeroth_dim = data_array.dims[0]

        # vessels to aggregate evaluated batches
        features_arrays = []

        with self.graph.as_default():
            # if True:  # to match indentation
            #     sess = tf.InteractiveSession(graph=self.graph)
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                logger.debug("running batches")
                length = data_array.sizes[zeroth_dim]
                for i in range(0, length, self.batch_size):
                    # handcrafted lazy feeding
                    sess.run(
                        self.iterator.initializer,
                        feed_dict={self.data: data_array[i:i+self.batch_size]}
                    )
                    features_batch_vec = sess.run([self.features])
                    batch_array = xr.DataArray(
                        data=features_batch_vec[0]
                    )
                    features_arrays.append(batch_array)
                logger.info('Encoding by tensorflow finished')

        features_array = (
            xr.concat(
                features_arrays, dim='dim_0'
            ).rename(
                {'dim_0': zeroth_dim, 'dim_1': 'hub_feature'}
            ).assign_coords(
                **{zeroth_dim: data_array.coords[zeroth_dim]}
            ).assign_attrs(self.attributes)
        )

        return features_array


class Flatten(Encoder):
    """Flatten a xr.DataArray over all dimensions but one."""

    def __init__(self, dim='features', dim_to_keep='', **kwargs):
        """
        Initialize the flattening encoder.

        Args:
            dim (str): name of the dimension generated by flattening, defaults
                to 'features'.
            dim_to_keep (str): name of the dimension to keep, defaults to ''
                that conists in flattening all dimensions but the first.
            kwargs (dict): arguments to pass to Encoder as attributes.
        """
        super(Flatten, self).__init__(attributes=kwargs)
        self.dim = dim
        self.dim_to_keep = dim_to_keep

    def __call__(self, data_array):
        """
        Encoding a xr.DataArray by flattening all dimensions but one.
        The kept dimension becomes the first of the generated xr.DataArray.

        Args:
            data_array (xr.DataArray): a data array that has to be
                flattened.
        Returns:
            a xr.DataArray.
        """
        dimensions = data_array.dims
        # handle the case where the dim is not provided
        if len(self.dim_to_keep) < 1:
            self.dim_to_keep = dimensions[0]
        # here we preserve the order of the dimensions for consistency
        # same can be achived with an OrderedSet but it seemed an
        # overkill
        to_flatten = [
            dimension
            for dimension in dimensions
            if dimension != self.dim_to_keep
        ]
        return data_array.stack(
            {self.dim: to_flatten}
        ).assign_attrs(self.attributes)
