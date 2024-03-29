﻿using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using KanjiClassificationWebAPI.Processors;
using Microsoft.Extensions.Logging;

namespace KanjiClassificationWebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClassificationController : ControllerBase
    {
        private readonly InferenceSession _session;
        private readonly Dictionary<int, string> _classDict;
        private readonly ILogger<ClassificationController> _logger;
        public ClassificationController(InferenceSession session, Dictionary<int, string> classDict, ILogger<ClassificationController> logger)
        {
            _session = session;
            _classDict = classDict;
            _logger = logger;
        }

        [HttpPost("ClassifyImage")]
        public IActionResult ClassifyImage(IFormFile imageBinary)
        {
            //Check for null or empty input
            if(imageBinary == null ||  imageBinary.Length == 0)
            {
                return BadRequest("No image was uploaded.");
            }

            //Validate image format
            if(!imageBinary.ContentType.Equals("image/png"))
            {
                return BadRequest("Invalid image format. Please upload a PNG image.");
            }

            try
            {

                _logger.LogInformation(imageBinary.ToString());

                var tensor = ImageProcessor.ProcessImage(imageBinary);

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input.1", tensor) };

                var outputs = _session.Run(inputs);

                var outputsArray = outputs.First().AsEnumerable<float>().ToArray();

                var maxValue = outputsArray.Max();

                var maxValueIndex = Array.IndexOf(outputsArray, maxValue);

                var predictedValue = _classDict[maxValueIndex];

                return Ok(predictedValue);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An error occurred while processing the image.");
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occurred while processing the image.");
            }
        }

        [HttpPost("InferTopKanjiClasses")]
        public IActionResult InferTopKanjiClasses(IFormFile imageBinary)
        {
            //Check for null or empty input
            if (imageBinary == null || imageBinary.Length == 0)
            {
                return BadRequest("No image was uploaded.");
            }

            //Validate image format
            if (!imageBinary.ContentType.Equals("image/png"))
            {
                return BadRequest("Invalid image format. Please upload a PNG image.");
            }

            try
            {

                var tensor = ImageProcessor.ProcessImage(imageBinary);

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input.1", tensor) };

                var outputs = _session.Run(inputs);

                var outputsArray = outputs.First().AsEnumerable<float>().ToArray();

                var top10ValuesWithIndices = outputsArray
                    .Select((value, index) => new { value, index })
                    .OrderByDescending(x => x.value)
                    .Take(10)
                    .ToArray();

                var topClasses = top10ValuesWithIndices.Select(x => _classDict[x.index]).ToArray();

                return Ok(topClasses);
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "An error occured while processing the image.");
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occured while processing the image.");
            }

        }
    }
}
