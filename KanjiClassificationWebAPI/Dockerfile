#See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
USER app
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["KanjiClassificationWebAPI/KanjiClassificationWebAPI.csproj", "KanjiClassificationWebAPI/"]
RUN dotnet restore "./KanjiClassificationWebAPI/./KanjiClassificationWebAPI.csproj"
RUN apt-get update && apt-get install -y libgomp1
COPY . .
WORKDIR "/src/KanjiClassificationWebAPI"
RUN dotnet build "./KanjiClassificationWebAPI.csproj" -c $BUILD_CONFIGURATION -o /app/build
WORKDIR "/src/KanjiClassificationWebAPI"
RUN ls -la


FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./KanjiClassificationWebAPI.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false
COPY ["KanjiClassificationWebAPI/kanjialive_kanjivg_classifier_v1.onnx", "/app/publish/"]
COPY ["KanjiClassificationWebAPI/class_dict.json", "/app/publish/"]
WORKDIR /src
RUN ls -la


FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
RUN ls -la
ENV ASPNETCORE_URLS http://*:8080
ENTRYPOINT ["dotnet", "KanjiClassificationWebAPI.dll"]