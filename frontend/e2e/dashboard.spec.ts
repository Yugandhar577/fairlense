import { expect, test } from "@playwright/test";

test("saved measurements, navigation, filters, exports, and mobile layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Baseline benchmark" }),
  ).toBeVisible();
  const results = await (await page.request.get("/api/results")).json();
  const baseline = results.performance.find(
    (r: any) =>
      r.model === "logistic_regression" && r.mitigation_state === "baseline",
  );
  await expect(
    page
      .getByRole("row")
      .filter({ hasText: "Logistic Regression" })
      .getByText(`${(baseline.accuracy * 100).toFixed(1)}%`, { exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/overview-desktop.png",
    fullPage: true,
  });

  await page
    .getByRole("link", { name: "Model performance", exact: true })
    .click();
  await page.getByLabel("Model", { exact: true }).selectOption("random_forest");
  await page.getByLabel("Experiment state").selectOption("reweighing");
  const mitigated = results.performance.find(
    (r: any) =>
      r.model === "random_forest" && r.mitigation_state === "reweighing",
  );
  await expect(
    page.getByText(`${(mitigated.accuracy * 100).toFixed(1)}%`, {
      exact: true,
    }),
  ).toBeVisible();

  await page
    .getByRole("link", { name: "Fairness analysis", exact: true })
    .click();
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue(
    "random_forest",
  );
  await page.getByLabel("Sensitive attribute").selectOption("gender_age_group");
  await expect(page.getByRole("table").getByRole("row")).toHaveCount(9);
  await page
    .getByLabel("Fairness metric")
    .selectOption("disparate_impact_ratio");

  await page
    .getByRole("link", { name: "Mitigation comparison", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Changes from baseline" }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "Cross-model comparison", exact: true })
    .click();
  await page.getByLabel("Performance metric").selectOption("recall");
  await expect(
    page.getByRole("heading", { name: "Recall across models" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "Recall versus Disparate impact" }),
  ).toBeVisible();

  await page
    .getByRole("link", { name: "Cross-group effects", exact: true })
    .click();
  await page
    .getByLabel("Fairness metric")
    .selectOption("equal_opportunity_diff");
  await expect(
    page.getByRole("heading", { name: "Cross-group comparison", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("table").getByRole("row")).toHaveCount(4);
  await page.screenshot({
    path: "test-results/effects-desktop.png",
    fullPage: true,
  });

  await page.locator(".download-menu summary").click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Mitigation changes" }).click();
  expect((await downloadPromise).suggestedFilename()).toBe(
    "fairness_changes.csv",
  );
  await page
    .getByRole("button", { name: "Refresh saved results", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Cross-group comparison", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Methodology", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Four definitions of fairness" }),
  ).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation menu" }).click();
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Baseline benchmark" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/overview-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("missing results explain how to recover and retry loads actual results", async ({
  page,
}) => {
  await page.route("**/api/results", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({
        message:
          "Run python run_pipeline.py to generate the experiment results, then refresh.",
      }),
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("python run_pipeline.py");
  await page.unroute("**/api/results");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(
    page.getByRole("heading", { name: "Baseline benchmark" }),
  ).toBeVisible();
});

test("undefined metrics and configured small-group warnings are preserved", async ({
  page,
}) => {
  const results = await (await page.request.get("/api/results")).json();
  for (const row of results.fairness) {
    if (
      row.model === "logistic_regression" &&
      row.mitigation_state === "baseline" &&
      row.sensitive_attribute === "gender"
    ) {
      row.disparate_impact_ratio = null;
      row.group_count = 5;
    }
  }
  results.config.low_sample_threshold = 12;
  await page.route("**/api/results", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(results),
    }),
  );
  await page.goto("/#fairness");
  await page
    .getByLabel("Fairness metric")
    .selectOption("disparate_impact_ratio");
  await expect(page.getByText("Undefined", { exact: true })).toBeVisible();
  await expect(page.getByText(/fewer than 12 test samples/)).toBeVisible();
});
