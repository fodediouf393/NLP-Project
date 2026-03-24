param(
    [Parameter(Mandatory = $true)]
    [string]$Text
)

python -m src.asrs_sum.pipeline.predict --text "$Text" --config "configs/config.yaml"