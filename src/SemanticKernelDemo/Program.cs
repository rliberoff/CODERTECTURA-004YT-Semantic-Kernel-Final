using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;

using Azure;
using Azure.AI.OpenAI;

using Encamina.Enmarcha.SemanticKernel.Abstractions;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.Plugins.Memory;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;

using CancellationTokenSource cancellationTokenSource = new();

var builder = Host.CreateApplicationBuilder(args);

builder.Configuration.SetBasePath(Directory.GetCurrentDirectory())
                     .AddJsonFile(@"appsettings.json", optional: false, reloadOnChange: true)
                     .AddJsonFile($@"appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)
                     .AddJsonFile($@"appsettings.{Environment.UserName}.json", optional: true, reloadOnChange: true)
                     .AddEnvironmentVariables();

// Add logging configuration...

builder.Services.AddLogging(loggingBuilder => loggingBuilder.AddConsole().AddDebug().SetMinimumLevel(LogLevel.Information));

// Add application configuration options...

builder.Services.AddOptions<AzureOpenAIOptions>()
                .BindConfiguration(nameof(AzureOpenAIOptions))
                .ValidateDataAnnotations()
                .ValidateOnStart();

builder.Services.AddOptions<SemanticKernelOptions>()
                .BindConfiguration(nameof(SemanticKernelOptions))
                .ValidateDataAnnotations()
                .ValidateOnStart();

// Add Semantic Kernel configuration... 

using var serviceProvider = builder.Services.BuildServiceProvider();

var azureOpenAIOptions = serviceProvider.GetRequiredService<IOptions<AzureOpenAIOptions>>().Value;
var semanticKernelOptions = serviceProvider.GetRequiredService<IOptions<SemanticKernelOptions>>().Value;

var oaiClientOptions = new OpenAIClientOptions(azureOpenAIOptions.ServiceVersion);
oaiClientOptions.Retry.MaxRetries = 3;
oaiClientOptions.Retry.NetworkTimeout = TimeSpan.FromMinutes(5);

var oaiClient = new OpenAIClient(semanticKernelOptions.Endpoint, new AzureKeyCredential(semanticKernelOptions.Key), oaiClientOptions);

var kernelBuilder = builder.Services.AddKernel()
                                    .AddAzureOpenAIChatCompletion(semanticKernelOptions.ChatModelDeploymentName, oaiClient)
                                    .AddAzureOpenAIChatCompletion(semanticKernelOptions.ChatModelDeploymentName, oaiClient, serviceId: @"service-gpt-4")
                                    ;

kernelBuilder.Plugins.AddFromPromptDirectory(Path.Combine(Values.PluginsDirectory, @"MiscPlugin"))
                     .AddFromPromptDirectory(Path.Combine(Values.PluginsDirectory, @"WriterPlugin"))
                     ;

builder.Services.AddSingleton(_ =>
{

#pragma warning disable SKEXP0003 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0011 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    return new Microsoft.SemanticKernel.Memory.MemoryBuilder()
            .WithAzureOpenAITextEmbeddingGeneration(semanticKernelOptions.EmbeddingsModelDeploymentName, semanticKernelOptions.EmbeddingsModelName, semanticKernelOptions.Endpoint.AbsoluteUri, semanticKernelOptions.Key)
            .WithMemoryStore(new VolatileMemoryStore())
            .Build();

#pragma warning restore SKEXP0011 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning restore SKEXP0003 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

});

// Add examples application and execute...

builder.Services.AddTransient<App>();

using var host = builder.Build();

try
{
    await host.Services.GetRequiredService<App>().RunAsync(args, cancellationTokenSource.Token);
}
catch (Exception e)
{
    Console.Error.WriteLine(e);
}

internal class App
{
    private readonly Kernel kernel;

    public App(Kernel kernel)
    {
        this.kernel = kernel;
    }

    public async Task RunAsync(string[] args, CancellationToken cancellationToken)
    {
        await ExampleA(cancellationToken);

        await ExampleB(cancellationToken);

        await ExampleC(cancellationToken);
    }

    private async Task ExampleA(CancellationToken cancellationToken)
    {
        var translateFunction = kernel.Plugins.GetFunction(@"WriterPlugin", @"Translate");

        Console.WriteLine(await translateFunction.InvokeAsync(kernel, new()
        {
            { @"input", @"Semantic Kernel is an SDK that integrates Large Language Models (LLMs) like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages like C#, Python, and Java. Semantic Kernel achieves this by allowing you to define plugins that can be chained together in just a few lines of code." },
            { @"language", @"Spanish" }
        }, cancellationToken));
    }

    private async Task ExampleB(CancellationToken cancellationToken)
    {
        var functionCreateCarnismRecipe = await GetKernelFunctionFromAssemblyAsync(Values.CreateCarnismRecipe, cancellationToken: cancellationToken);
        var functionCreateVeganRecipe = await GetKernelFunctionFromAssemblyAsync(Values.CreateVeganRecipe, cancellationToken: cancellationToken);
        var functionCreateVeganRecipeVariousIngredients = await GetKernelFunctionFromAssemblyAsync(Values.CreateVeganRecipeVariousIngredients, new HandlebarsPromptTemplateFactory(), cancellationToken: cancellationToken);

        Console.WriteLine(await functionCreateCarnismRecipe.InvokeAsync(kernel, new() { { @"input", @"Main" } }, cancellationToken));
        Console.WriteLine(await functionCreateVeganRecipe.InvokeAsync(kernel, new() { { @"input", @"Dessert" } }, cancellationToken));
        Console.WriteLine(await functionCreateVeganRecipeVariousIngredients.InvokeAsync(kernel, new()
        {
            { @"input", @"Starter" },
            { @"ingredients", new[] { @"tomato", @"potato", @"eggplant" } }
        }, cancellationToken));
    }

    private async Task ExampleC(CancellationToken cancellationToken)
    {
        var functionCreateCarnismRecipe = await GetKernelFunctionFromAssemblyAsync(Values.CreateCarnismRecipe, cancellationToken: cancellationToken);
        var functionCreateVeganRecipe = await GetKernelFunctionFromAssemblyAsync(Values.CreateVeganRecipe, cancellationToken: cancellationToken);

        var plugin = new MutableKernelPlugin(@"Recipes");
        plugin.AddFunction(functionCreateCarnismRecipe);
        plugin.AddFunction(functionCreateVeganRecipe);

        kernel.Plugins.Add(plugin);

#pragma warning disable SKEXP0060 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
        
        var planner = new HandlebarsPlanner(new HandlebarsPlannerOptions()
        {
            AllowLoops = true
        });

        var plan = await planner.CreatePlanAsync(kernel, @"Create a book of vegan recipes with 3 recipes per chapter, first chapter for starters, second chapter for main courses, and final chapter for desserts. The book is called 'Vegan recipes of my Grandma'.");

        Stopwatch stopwatch = new();
        stopwatch.Start();

        var planResult = await plan.InvokeAsync(kernel, new(), cancellationToken);

        stopwatch.Stop();

        Console.WriteLine(planResult);
        Console.WriteLine($@"Plan execution took {stopwatch.ElapsedMilliseconds} ms.");

#pragma warning restore SKEXP0060 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }

    private async Task<KernelFunction> GetKernelFunctionFromAssemblyAsync(string propmtFunctionResourceName, IPromptTemplateFactory? promptTemplateFactory = null, CancellationToken cancellationToken = default)
    {
        using StreamReader reader = new(Assembly.GetExecutingAssembly().GetManifestResourceStream(propmtFunctionResourceName)!);
        return kernel.CreateFunctionFromPromptYaml(await reader.ReadToEndAsync(cancellationToken), promptTemplateFactory);
    }
}

internal sealed class AzureOpenAIOptions
{
    public OpenAIClientOptions.ServiceVersion ServiceVersion { get; set; } = OpenAIClientOptions.ServiceVersion.V2023_12_01_Preview;
}

internal sealed class MutableKernelPlugin : KernelPlugin
{
    private readonly Dictionary<string, KernelFunction> functions;

    public MutableKernelPlugin(string name, string? description = null, IEnumerable<KernelFunction>? functions = null) : base(name, description)
    {
        this.functions = new Dictionary<string, KernelFunction>(StringComparer.OrdinalIgnoreCase);

        if (functions is not null)
        {
            foreach (var f in functions)
            {
                ArgumentNullException.ThrowIfNull(f);

                this.functions.Add(f.Name, f);
            }
        }
    }

    /// <inheritdoc/>
    public override int FunctionCount => functions.Count;

    /// <inheritdoc/>
    public override bool TryGetFunction(string name, [NotNullWhen(true)] out KernelFunction? function) => functions.TryGetValue(name, out function);

    public void AddFunction(KernelFunction function)
    {
        ArgumentNullException.ThrowIfNull(function);

        functions.Add(function.Name, function);
    }

    /// <inheritdoc/>
    public override IEnumerator<KernelFunction> GetEnumerator() => functions.Values.GetEnumerator();
}

internal static class Values
{
    public static readonly string PluginsDirectory = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, @"Plugins");

    public const string CreateCarnismRecipe = @"SemanticKernelDemo.Prompts.CreateCarnismRecipe.yaml";

    public const string CreateVeganRecipe = @"SemanticKernelDemo.Prompts.CreateVeganRecipe.yaml";

    public const string CreateVeganRecipeVariousIngredients = @"SemanticKernelDemo.Prompts.CreateVeganRecipeVariousIngredients.yaml";
}
