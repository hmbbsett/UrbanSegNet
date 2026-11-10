import argparse
import csv
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def evaluate(maps_dir, reference_dir):
    with (reference_dir / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2000 or len({row['filename'] for row in rows}) != 2000:
        raise ValueError('Expected the final manifest of 2,000 distinct reference masks.')

    all_cities = np.zeros((8, 9), dtype=np.int64)
    comparison = np.zeros_like(all_cities)
    city_key = itemgetter('city')
    cities = 0
    comparison_patches = 0
    for city, patches in groupby(sorted(rows, key=city_key), key=city_key):
        cities += 1
        with rasterio.open(maps_dir / f'{city}.tif') as product:
            for patch in patches:
                name = patch['filename']
                window = Window(
                    int(patch['col0_in_city_product']),
                    int(patch['row0_in_city_product']),
                    100,
                    100,
                )
                with rasterio.open(reference_dir / 'GT' / name) as reference:
                    if (
                        reference.shape != (100, 100)
                        or reference.crs != product.crs
                        or not reference.transform.almost_equals(product.window_transform(window))
                    ):
                        raise ValueError(f'{name}: reference grid does not match the city raster window.')
                    gt = reference.read(1)
                pred = product.read(1, window=window)
                if pred.shape != gt.shape or not np.all((gt >= 1) & (gt <= 8)):
                    raise ValueError(f'{name}: incomplete window or invalid reference class.')
                pred = np.where((pred >= 1) & (pred <= 8), pred, 0)
                pairs = (gt.astype(np.int64) - 1) * 9 + pred
                counts = np.bincount(pairs.ravel(), minlength=72).reshape(8, 9)
                all_cities += counts
                if city != 'Guiyang':
                    comparison += counts
                    comparison_patches += 1

    if cities != 31 or comparison_patches != 1973:
        raise ValueError('Expected 31 cities, including 27 Guiyang masks.')
    return [(31, len(rows), all_cities), (30, comparison_patches, comparison)]


def metrics(counts):
    correct = counts[:, 1:].diagonal()
    reference = counts.sum(axis=1)
    predicted = counts[:, 1:].sum(axis=0)
    union = reference + predicted - correct

    def macro(numerator, denominator):
        defined = denominator > 0
        return 100 * np.mean(numerator[defined] / denominator[defined])

    return (
        100 * correct.sum() / counts.sum(),
        macro(correct, reference),
        macro(correct, predicted),
        macro(2 * correct, reference + predicted),
        macro(correct, union),
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate the final CHRUC-2024 city GeoTIFFs.')
    parser.add_argument('--maps-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    results = evaluate(args.maps_dir, args.reference_dir)
    print('Cities,Patches,Reference pixels,OA (%),mPA (%),mUA (%),mF1 (%),mIoU (%)')
    for cities, patches, counts in results:
        values = ','.join(f'{value:.2f}' for value in metrics(counts))
        print(f'{cities},{patches},{counts.sum()},{values}')


if __name__ == '__main__':
    main()
