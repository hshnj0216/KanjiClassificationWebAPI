using Microsoft.ML.OnnxRuntime;
using Microsoft.ML;
using System.Text.Json;
using Microsoft.Extensions.Logging;

var builder = WebApplication.CreateBuilder(args);

var modelPath = builder.Configuration["ModelPath"];
var dictionaryPath = builder.Configuration["ClassDictionaryPath"];


// Add services to the container.
builder.Services.AddSingleton(provider => new InferenceSession(modelPath));
builder.Services.AddSingleton(provider =>
{
    var jsonString = File.ReadAllText(dictionaryPath);
    Dictionary<int, string> classDict = JsonSerializer.Deserialize<Dictionary<int, string>>(jsonString);
    return classDict;
});

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

//app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
