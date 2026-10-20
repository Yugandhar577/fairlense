import { expect, test } from "@playwright/test";

const csv =
  "value,category,group,age_band,outcome,price\n" +
  Array.from(
    { length: 180 },
    (_, i) =>
      `${i / 10},${i % 3 ? "urban" : "rural"},${i % 2 ? "B" : "A"},${i % 5 ? "younger" : "older"},${i % 4 < 2 ? "no" : "yes"},${i * 2 + (i % 3)}`,
  ).join("\n");

test("upload, train two classifiers with mitigation, inspect saved results and export", async ({
  page,
}) => {
  const errors: string[] = [];
  const datasetName = `classification-demo-${Date.now()}.csv`;
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#train");
  await expect(
    page.getByRole("heading", { name: "Dataset", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Upload dataset")).toBeEnabled();
  await page.getByLabel("Upload dataset").setInputFiles({
    name: datasetName,
    mimeType: "text/csv",
    buffer: Buffer.from(csv),
  });
  await page
    .getByLabel("Target column", { exact: true })
    .selectOption("outcome");
  await page.getByLabel("Positive / focus class").selectOption("yes");
  await page
    .getByRole("group", { name: "Sensitive columns (optional, up to 3)" })
    .getByRole("checkbox", { name: "group 2 groups", exact: true })
    .check();
  await page
    .getByRole("group", { name: "Sensitive columns (optional, up to 3)" })
    .getByRole("checkbox", { name: "age_band 2 groups", exact: true })
    .check();
  await page.getByLabel("Also evaluate the intersection").check();
  await page.getByLabel("Number of models").selectOption("2");
  await page
    .getByLabel("Model 2", { exact: true })
    .selectOption("decision_tree");
  await page.getByLabel("Reweighing", { exact: true }).check();
  await page.getByLabel("Threshold adjustment", { exact: false }).check();
  await page.screenshot({
    path: "test-results/training-classification-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Train 2 models", exact: true })
    .click();
  const run = page
    .getByRole("region", { name: "Experiment history" })
    .getByRole("row")
    .filter({ hasText: datasetName })
    .first();
  await expect(run).toContainText("completed", { timeout: 30000 });
  await run.getByRole("button", { name: "View results" }).click();
  await expect(
    page.getByRole("heading", { name: "Baseline benchmark", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Baseline benchmark" }).getByRole("row"),
  ).toHaveCount(3);
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    for (const route of [
      "overview",
      "performance",
      "fairness",
      "mitigation",
      "models",
      "effects",
      "methodology",
    ]) {
      await page.goto(`/#${route}`);
      await expect(page.locator(".page-content")).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
  }
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page
    .getByRole("link", { name: "Fairness analysis", exact: true })
    .click();
  await page.getByLabel("Sensitive attribute").selectOption("group × age_band");
  await expect(
    page
      .getByRole("region", { name: "Group-level measurements" })
      .getByRole("row"),
  ).toHaveCount(5);
  await page.screenshot({
    path: "test-results/training-classification-fairness.png",
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Cross-group effects", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Cross-group comparison", exact: true }),
  ).toBeVisible();
  await page.locator(".download-menu summary").click();
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Run configuration", exact: true })
    .click();
  expect((await downloading).suggestedFilename()).toBe("run_config.json");
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Cross-group comparison", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Results to view").selectOption("");
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Baseline benchmark", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("regression offers compatible models, uses regression metrics, and works on mobile", async ({
  page,
}) => {
  const datasetName = `regression-demo-${Date.now()}.csv`;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#train");
  await expect(page.getByLabel("Upload dataset")).toBeEnabled();
  await page.getByLabel("Upload dataset").setInputFiles({
    name: datasetName,
    mimeType: "text/csv",
    buffer: Buffer.from(csv),
  });
  await page.getByLabel("Task type").selectOption("regression");
  await page.getByLabel("Target column", { exact: true }).selectOption("price");
  await page.getByLabel("Number of models").selectOption("2");
  await page.getByLabel("Model 2", { exact: true }).selectOption("ridge");
  await expect(page.getByLabel("Model 1", { exact: true })).toHaveValue(
    "linear_regression",
  );
  await expect(
    page
      .getByLabel("Model 1", { exact: true })
      .getByRole("option", { name: "Logistic Regression", exact: true }),
  ).toHaveCount(0);
  await page.screenshot({
    path: "test-results/training-regression-form-mobile.png",
    fullPage: true,
  });
  await page
    .getByRole("group", { name: "Sensitive columns (optional, up to 3)" })
    .getByRole("checkbox", { name: "group 2 groups", exact: true })
    .check();
  await expect(page.getByLabel("Reweighing", { exact: true })).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page
    .getByRole("button", { name: "Train 2 models", exact: true })
    .click();
  const run = page
    .getByRole("region", { name: "Experiment history" })
    .getByRole("row")
    .filter({ hasText: datasetName })
    .first();
  await expect(run).toContainText("completed", { timeout: 30000 });
  await run.getByRole("button", { name: "View results" }).click();
  await expect(
    page.getByRole("columnheader", { name: "RMSE", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Accuracy", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Open navigation menu" }).click();
  await page
    .getByRole("link", { name: "Fairness analysis", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Group-level regression measurements" }),
  ).toBeVisible();
  await expect(
    page.getByText("These are regression group-error diagnostics", {
      exact: false,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/training-regression-mobile.png",
    fullPage: true,
  });
  for (const route of [
    "overview",
    "performance",
    "mitigation",
    "models",
    "effects",
    "methodology",
  ]) {
    await page.goto(`/#${route}`);
    await expect(page.locator(".page-content")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});

test("invalid uploads and cross-task model choices fail clearly without starting a run", async ({
  page,
}) => {
  await page.goto("/#train");
  await expect(page.getByLabel("Upload dataset")).toBeEnabled();
  await page.getByLabel("Upload dataset").setInputFiles({
    name: "invalid.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("a,b\n1,2\n"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "between 20 and 50,000 rows",
  );
  const uploaded = await page.request.post("/api/workspace/datasets", {
    data: { name: "validation.csv", csv },
  });
  expect(uploaded.status()).toBe(201);
  const profile = await uploaded.json();
  const response = await page.request.post("/api/workspace/runs", {
    data: {
      dataset_id: profile.id,
      task: "classification",
      target: "outcome",
      positive_label: "yes",
      features: ["value"],
      sensitive: [],
      models: ["linear_regression", "logistic_regression"],
    },
  });
  expect(response.status()).toBe(422);
  expect((await response.json()).message).toContain(
    "must support the chosen task",
  );
  const unsafe = await page.request.post("/api/workspace/datasets", {
    headers: { Origin: "https://unrelated.example" },
    data: { name: "unwanted.csv", csv },
  });
  expect(unsafe.status()).toBe(403);
});

test("training workspace does not require the original Adult results", async ({
  page,
}) => {
  await page.route("**/api/results", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({
        message: "Original Adult results have not been generated.",
      }),
    }),
  );
  await page.goto("/#train");
  await expect(page.getByLabel("Upload dataset")).toBeEnabled();
  await expect(
    page.getByRole("heading", { name: "Dataset", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Experiment results unavailable" }),
  ).toHaveCount(0);
});
