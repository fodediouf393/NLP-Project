param(
    [Parameter(Mandatory = $true)]
    [string]$Input,

    [Parameter(Mandatory = $true)]
    [string]$Output
)

python -m src.asrs_sum.evaluation.evaluate --input $Input --output $Output --config "configs/config.yaml"