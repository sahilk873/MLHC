# Results Snapshot

## Occult Hypoxemia Classifier (BOLD Test)
{
  "auroc": 0.6469380919670928,
  "auprc": 0.03670453722266454,
  "precision": 0.03156872268091306,
  "recall": 0.8369098712446352,
  "prevalence": 0.02376338602753697,
  "lift": 1.544583637202686,
  "subgroup_fnr": {
    "American Indian / Alaska Native": {
      "fnr": NaN,
      "tp": 0,
      "fn": 0,
      "positives": 0
    },
    "Asian": {
      "fnr": 0.0,
      "tp": 1,
      "fn": 0,
      "positives": 1
    },
    "Black": {
      "fnr": 0.09523809523809523,
      "tp": 38,
      "fn": 4,
      "positives": 42
    },
    "Hispanic OR Latino": {
      "fnr": 0.2,
      "tp": 8,
      "fn": 2,
      "positives": 10
    },
    "Unknown": {
      "fnr": 0.35714285714285715,
      "tp": 9,
      "fn": 5,
      "positives": 14
    },
    "White": {
      "fnr": 0.16265060240963855,
      "tp": 139,
      "fn": 27,
      "positives": 166
    }
  },
  "subgroup_fnr_ci": {
    "Black": {
      "mean": 0.09436900139686469,
      "ci_low": 0.0,
      "ci_high": 0.15789473684210525
    },
    "White": {
      "mean": 0.1622839263470844,
      "ci_low": 0.11764705882352941,
      "ci_high": 0.20354392002622093
    }
  }
}

## Conformal Width Stats (Global)
{
  "coverage_90": 0.8963793982661907,
  "mean_width_90": 8.721035007610368,
  "median_width_90": 8.721035007610368,
  "group_coverage_90": {
    "American Indian / Alaska Native": 0.9193548387096774,
    "Asian": 0.949438202247191,
    "Black": 0.8677685950413223,
    "Hispanic OR Latino": 0.9187358916478555,
    "Unknown": 0.8927576601671309,
    "White": 0.8976600322754169
  },
  "group_width_90": {
    "American Indian / Alaska Native": 8.721035007610364,
    "Asian": 8.721035007610368,
    "Black": 8.721035007610368,
    "Hispanic OR Latino": 8.721035007610368,
    "Unknown": 8.721035007610368,
    "White": 8.721035007610368
  },
  "group_width_median_90": {
    "American Indian / Alaska Native": 8.721035007610368,
    "Asian": 8.721035007610368,
    "Black": 8.721035007610368,
    "Hispanic OR Latino": 8.721035007610368,
    "Unknown": 8.721035007610368,
    "White": 8.721035007610368
  },
  "coverage_95": 0.9482916879143294,
  "mean_width_95": 13.11554404145079,
  "median_width_95": 13.11554404145079,
  "group_coverage_95": {
    "American Indian / Alaska Native": 0.9354838709677419,
    "Asian": 0.9662921348314607,
    "Black": 0.9297520661157025,
    "Hispanic OR Latino": 0.9503386004514672,
    "Unknown": 0.9484679665738162,
    "White": 0.950242065626681
  },
  "group_width_95": {
    "American Indian / Alaska Native": 13.11554404145079,
    "Asian": 13.115544041450791,
    "Black": 13.115544041450791,
    "Hispanic OR Latino": 13.115544041450791,
    "Unknown": 13.115544041450791,
    "White": 13.115544041450791
  },
  "group_width_median_95": {
    "American Indian / Alaska Native": 13.11554404145079,
    "Asian": 13.11554404145079,
    "Black": 13.11554404145079,
    "Hispanic OR Latino": 13.11554404145079,
    "Unknown": 13.11554404145079,
    "White": 13.11554404145079
  }
}

## Mondrian Conformal (Race-conditional)
{
  "group_coverage_90": {
    "American Indian / Alaska Native": 0.9193548387096774,
    "Asian": 0.949438202247191,
    "Black": 0.8946280991735537,
    "Hispanic OR Latino": 0.909706546275395,
    "Unknown": 0.871866295264624,
    "White": 0.8944324905863368
  },
  "group_width_90": {
    "American Indian / Alaska Native": 9.857629604001684,
    "Asian": 8.88779193954673,
    "Black": 10.504125306500212,
    "Hispanic OR Latino": 8.440339657605364,
    "Unknown": 7.576207333311686,
    "White": 8.55776720034612
  },
  "group_width_median_90": {
    "American Indian / Alaska Native": 9.857629604001687,
    "Asian": 8.88779193954673,
    "Black": 10.504125306500214,
    "Hispanic OR Latino": 8.440339657605364,
    "Unknown": 7.576207333311686,
    "White": 8.55776720034612
  },
  "group_coverage_95": {
    "American Indian / Alaska Native": 0.9354838709677419,
    "Asian": 0.9719101123595506,
    "Black": 0.9566115702479339,
    "Hispanic OR Latino": 0.9480812641083521,
    "Unknown": 0.9484679665738162,
    "White": 0.9488972565895643
  },
  "group_width_95": {
    "American Indian / Alaska Native": 12.322509859818638,
    "Asian": 14.889489489489362,
    "Black": 17.185962860234554,
    "Hispanic OR Latino": 12.440328683815068,
    "Unknown": 12.97249061913706,
    "White": 12.884455958549209
  },
  "group_width_median_95": {
    "American Indian / Alaska Native": 12.322509859818638,
    "Asian": 14.889489489489364,
    "Black": 17.185962860234554,
    "Hispanic OR Latino": 12.440328683815068,
    "Unknown": 12.972490619137062,
    "White": 12.88445595854921
  }
}