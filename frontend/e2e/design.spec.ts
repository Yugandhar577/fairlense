import { expect, test } from "@playwright/test";

const routes = [
  "overview",
  "performance",
  "fairness",
  "mitigation",
  "models",
  "effects",
  "methodology",
];

test("all screens keep their workflow and readable layout at desktop, tablet, and mobile widths", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    for (const route of routes) {
      await page.goto(`/#${route}`);
      await expect(page.locator(".page-content")).toBeVisible();
      await expect(page.locator("main h1")).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      // Chrome's normal chart labels are SVG; application copy must stay readable.
      const smallCopy = await page
        .locator("main p, main dt, main td, main th, .select-field > span")
        .evaluateAll((elements) =>
          elements
            .filter(
              (element) =>
                element.getBoundingClientRect().width > 0 &&
                Number.parseFloat(getComputedStyle(element).fontSize) < 13,
            )
            .map((element) => element.textContent),
        );
      expect(smallCopy).toEqual([]);
      await expect(
        page.locator(
          ".avatar, .workspace-avatar, .status-dot, .study-card, .stat-card, .model-index, .eyebrow",
        ),
      ).toHaveCount(0);
      if (route === "overview") {
        await expect(
          page.getByRole("heading", { name: "Experiment configuration" }),
        ).toBeVisible();
        const headings = await page.locator("main h2").allTextContents();
        expect(headings[0]).toBe("Baseline benchmark");
      }
    }
  }
  expect(errors).toEqual([]);
});

test("mobile navigation traps focus, supports Escape, and restores focus", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const opener = page.getByRole("button", { name: "Open navigation menu" });
  await opener.click();
  const closer = page
    .getByRole("complementary", { name: "Workspace navigation" })
    .getByRole("button", { name: "Close navigation menu" });
  await expect(closer).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(
    page.getByRole("link", { name: "Methodology", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(closer).toBeFocused();
  await expect(page.locator(".main-shell")).toHaveAttribute("inert", "");
  await page.keyboard.press("Escape");
  await expect(opener).toBeFocused();
  await expect(page.locator(".main-shell")).not.toHaveAttribute("inert");
  await expect(opener).toHaveAttribute("aria-expanded", "false");
});

test("scatter measurements are available without hover and track the selected view", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Sensitive attribute").selectOption("age_group");
  await page
    .getByLabel("Fairness metric")
    .selectOption("equal_opportunity_diff");
  await page.getByText("View plotted measurements", { exact: true }).click();
  const table = page
    .getByRole("region", { name: "Plotted measurements", exact: true })
    .getByRole("table");
  const results = await (await page.request.get("/api/results")).json();
  const expected = results.fairness.find(
    (row: any) =>
      row.model === "logistic_regression" &&
      row.mitigation_state === "baseline" &&
      row.sensitive_attribute === "age_group",
  );
  await expect(
    table.getByRole("columnheader", { name: "Equal opportunity" }),
  ).toBeVisible();
  await expect(
    table
      .getByRole("row")
      .filter({ hasText: "Logistic Regression" })
      .filter({ hasText: "Baseline" }),
  ).toContainText(expected.equal_opportunity_diff.toFixed(3));
  await expect(table.getByRole("row")).toHaveCount(10);
});

test("reduced motion has no decorative page animations or glass and gradient surfaces", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#methodology");
  await expect(
    page.getByRole("heading", { name: "Experimental protocol" }),
  ).toBeVisible();
  const effects = await page
    .locator(
      ".app-shell, .sidebar, main, .page-content, .panel, .metric-value, button",
    )
    .evaluateAll((elements) =>
      elements.map((element) => {
        const style = getComputedStyle(element);
        return {
          animation: style.animationName,
          backdrop: style.backdropFilter,
          gradient: /gradient\(/.test(style.backgroundImage),
        };
      }),
    );
  for (const effect of effects) {
    expect(effect.animation).toBe("none");
    expect(effect.backdrop).toBe("none");
    expect(effect.gradient).toBe(false);
  }
  expect(
    await page
      .locator(".protocol")
      .evaluate((element) => getComputedStyle(element).display),
  ).toBe("flex");
});
