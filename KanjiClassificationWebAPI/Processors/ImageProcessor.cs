using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace KanjiClassificationWebAPI.Processors
{
    public class ImageProcessor
    {
        //Converts image binary to a tensor
        public static DenseTensor<float> ProcessImage(IFormFile imageBinary)
        {
            using var imageStream = new MemoryStream();
            imageBinary.CopyTo(imageStream);
            imageStream.Position = 0;

            using var image = Image.Load<Rgb24>(imageStream);

            image.Mutate(x => x.Resize(new ResizeOptions { Size = new Size(224, 224), Mode = ResizeMode.Crop}));

            var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, 0, y, x] = pixel.R / 255.0f;
                    tensor[0, 1, y, x] = pixel.G / 255.0f;
                    tensor[0, 2, y, x] = pixel.B / 255.0f;
                }
            }

            return tensor;

        }
    }
}
