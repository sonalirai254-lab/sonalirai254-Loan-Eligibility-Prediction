import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Loan Eligibility Predictor" },
      {
        name: "description",
        content:
          "Estimate your loan eligibility instantly based on income, credit score, employment, and loan details.",
      },
      { property: "og:title", content: "Loan Eligibility Predictor" },
      {
        property: "og:description",
        content:
          "Estimate your loan eligibility instantly based on income, credit score, employment, and loan details.",
      },
    ],
  }),
  component: Index,
});

type FormState = {
  fullName: string;
  age: number;
  monthlyIncome: number;
  monthlyExpenses: number;
  creditScore: number;
  employmentYears: number;
  employmentType: "salaried" | "self-employed" | "unemployed";
  loanAmount: number;
  loanTermYears: number;
  existingEmis: number;
  hasCollateral: boolean;
};

const defaultForm: FormState = {
  fullName: "",
  age: 30,
  monthlyIncome: 60000,
  monthlyExpenses: 20000,
  creditScore: 720,
  employmentYears: 4,
  employmentType: "salaried",
  loanAmount: 500000,
  loanTermYears: 5,
  existingEmis: 0,
  hasCollateral: false,
};

function calcEMI(principal: number, annualRate: number, years: number) {
  const r = annualRate / 12 / 100;
  const n = years * 12;
  if (r === 0) return principal / n;
  return (principal * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
}

type Result = {
  eligible: boolean;
  score: number;
  estimatedEMI: number;
  maxLoan: number;
  interestRate: number;
  reasons: string[];
};

function predict(f: FormState): Result {
  const reasons: string[] = [];
  let score = 0;

  // Credit score (0-35)
  if (f.creditScore >= 750) score += 35;
  else if (f.creditScore >= 700) score += 28;
  else if (f.creditScore >= 650) score += 18;
  else if (f.creditScore >= 600) score += 8;
  else reasons.push("Credit score below 600 significantly reduces eligibility.");

  // Income vs expenses (0-20)
  const disposable = f.monthlyIncome - f.monthlyExpenses - f.existingEmis;
  if (disposable <= 0) {
    reasons.push("Monthly expenses and existing EMIs exceed your income.");
  } else if (disposable > 40000) score += 20;
  else if (disposable > 20000) score += 15;
  else if (disposable > 10000) score += 10;
  else score += 4;

  // Employment (0-20)
  if (f.employmentType === "unemployed") {
    reasons.push("Stable employment is required for loan approval.");
  } else {
    if (f.employmentYears >= 5) score += 20;
    else if (f.employmentYears >= 2) score += 14;
    else if (f.employmentYears >= 1) score += 8;
    else score += 3;
    if (f.employmentType === "self-employed") score -= 3;
  }

  // Age (0-10)
  if (f.age >= 23 && f.age <= 55) score += 10;
  else if (f.age > 55 && f.age <= 60) score += 5;
  else reasons.push("Age is outside the typical lending range (23–60).");

  // Interest rate based on credit
  let interestRate = 14;
  if (f.creditScore >= 780) interestRate = 8.5;
  else if (f.creditScore >= 740) interestRate = 9.5;
  else if (f.creditScore >= 700) interestRate = 10.75;
  else if (f.creditScore >= 650) interestRate = 12.5;
  if (f.hasCollateral) interestRate -= 1;

  const estimatedEMI = calcEMI(f.loanAmount, interestRate, f.loanTermYears);

  // EMI-to-income (DTI) (0-15)
  const dti = (estimatedEMI + f.existingEmis) / Math.max(f.monthlyIncome, 1);
  if (dti <= 0.35) score += 15;
  else if (dti <= 0.5) score += 8;
  else {
    score += 0;
    reasons.push(
      `Debt-to-income ratio is ${(dti * 100).toFixed(0)}% — lenders prefer under 50%.`
    );
  }

  if (f.hasCollateral) score += 5;

  score = Math.max(0, Math.min(100, Math.round(score)));

  // Max loan: 50% of monthly income goes to EMI
  const maxEMI = Math.max(0, f.monthlyIncome * 0.5 - f.existingEmis);
  const r = interestRate / 12 / 100;
  const n = f.loanTermYears * 12;
  const maxLoan =
    r === 0 ? maxEMI * n : (maxEMI * (Math.pow(1 + r, n) - 1)) / (r * Math.pow(1 + r, n));

  const eligible =
    score >= 60 &&
    f.employmentType !== "unemployed" &&
    disposable > 0 &&
    estimatedEMI <= maxEMI;

  if (eligible && reasons.length === 0) {
    reasons.push("Strong credit profile and healthy debt-to-income ratio.");
  }
  if (!eligible && estimatedEMI > maxEMI) {
    reasons.push(
      `Requested EMI ₹${Math.round(estimatedEMI).toLocaleString()} exceeds safe limit ₹${Math.round(maxEMI).toLocaleString()}.`
    );
  }

  return {
    eligible,
    score,
    estimatedEMI,
    maxLoan: Math.max(0, Math.floor(maxLoan)),
    interestRate,
    reasons,
  };
}

function Index() {
  const [form, setForm] = useState<FormState>(defaultForm);
  const [submitted, setSubmitted] = useState(false);

  const result = useMemo(() => predict(form), [form]);

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((p) => ({ ...p, [key]: value }));

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold">
              ₹
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">LoanLens</h1>
              <p className="text-xs text-muted-foreground">Eligibility Predictor</p>
            </div>
          </div>
          <span className="text-xs text-muted-foreground">Instant estimate · No signup</span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        <section className="mb-8">
          <h2 className="text-3xl font-bold tracking-tight">
            Check your loan eligibility in seconds
          </h2>
          <p className="mt-2 text-muted-foreground">
            Fill in your financial details below. We estimate your eligibility, EMI, and the
            maximum loan you could qualify for using a transparent rule-based model.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-5">
          <form
            className="lg:col-span-3 space-y-5 rounded-xl border border-border bg-card p-6 shadow-sm"
            onSubmit={(e) => {
              e.preventDefault();
              setSubmitted(true);
            }}
          >
            <Field label="Full name">
              <input
                type="text"
                value={form.fullName}
                onChange={(e) => update("fullName", e.target.value)}
                placeholder="Jane Doe"
                className={inputCls}
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Age">
                <input
                  type="number"
                  min={18}
                  max={75}
                  value={form.age}
                  onChange={(e) => update("age", +e.target.value)}
                  className={inputCls}
                />
              </Field>
              <Field label="Credit score (300–850)">
                <input
                  type="number"
                  min={300}
                  max={850}
                  value={form.creditScore}
                  onChange={(e) => update("creditScore", +e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Monthly income (₹)">
                <input
                  type="number"
                  min={0}
                  value={form.monthlyIncome}
                  onChange={(e) => update("monthlyIncome", +e.target.value)}
                  className={inputCls}
                />
              </Field>
              <Field label="Monthly expenses (₹)">
                <input
                  type="number"
                  min={0}
                  value={form.monthlyExpenses}
                  onChange={(e) => update("monthlyExpenses", +e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Employment type">
                <select
                  value={form.employmentType}
                  onChange={(e) =>
                    update("employmentType", e.target.value as FormState["employmentType"])
                  }
                  className={inputCls}
                >
                  <option value="salaried">Salaried</option>
                  <option value="self-employed">Self-employed</option>
                  <option value="unemployed">Unemployed</option>
                </select>
              </Field>
              <Field label="Years employed">
                <input
                  type="number"
                  min={0}
                  max={50}
                  value={form.employmentYears}
                  onChange={(e) => update("employmentYears", +e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Loan amount (₹)">
                <input
                  type="number"
                  min={10000}
                  value={form.loanAmount}
                  onChange={(e) => update("loanAmount", +e.target.value)}
                  className={inputCls}
                />
              </Field>
              <Field label="Loan term (years)">
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={form.loanTermYears}
                  onChange={(e) => update("loanTermYears", +e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>

            <Field label="Existing EMIs / month (₹)">
              <input
                type="number"
                min={0}
                value={form.existingEmis}
                onChange={(e) => update("existingEmis", +e.target.value)}
                className={inputCls}
              />
            </Field>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.hasCollateral}
                onChange={(e) => update("hasCollateral", e.target.checked)}
                className="h-4 w-4 rounded border-input"
              />
              I can provide collateral (property, FD, etc.)
            </label>

            <div className="flex gap-3 pt-2">
              <button
                type="submit"
                className="inline-flex items-center justify-center rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Predict eligibility
              </button>
              <button
                type="button"
                onClick={() => {
                  setForm(defaultForm);
                  setSubmitted(false);
                }}
                className="inline-flex items-center justify-center rounded-md border border-input bg-background px-5 py-2.5 text-sm font-medium transition-colors hover:bg-accent"
              >
                Reset
              </button>
            </div>
          </form>

          <aside className="lg:col-span-2">
            <div className="sticky top-6 rounded-xl border border-border bg-card p-6 shadow-sm">
              <h3 className="text-sm font-medium text-muted-foreground">
                {submitted ? "Prediction result" : "Live estimate"}
              </h3>

              <div
                className={`mt-3 inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ${
                  result.eligible
                    ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                    : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    result.eligible ? "bg-green-600" : "bg-red-600"
                  }`}
                />
                {result.eligible ? "Likely Eligible" : "Not Eligible"}
              </div>

              <div className="mt-5">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-muted-foreground">Eligibility score</span>
                  <span className="text-2xl font-bold">{result.score}</span>
                </div>
                <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className={`h-full transition-all ${
                      result.score >= 60 ? "bg-green-600" : "bg-red-500"
                    }`}
                    style={{ width: `${result.score}%` }}
                  />
                </div>
              </div>

              <dl className="mt-5 space-y-3 text-sm">
                <Row label="Estimated EMI">
                  ₹{Math.round(result.estimatedEMI).toLocaleString()}/mo
                </Row>
                <Row label="Interest rate (est.)">{result.interestRate.toFixed(2)}%</Row>
                <Row label="Max loan you could get">
                  ₹{result.maxLoan.toLocaleString()}
                </Row>
              </dl>

              <div className="mt-5">
                <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Why
                </h4>
                <ul className="mt-2 space-y-1.5 text-sm">
                  {result.reasons.map((r, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="text-muted-foreground">•</span>
                      <span>{r}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <p className="mt-5 text-[11px] leading-relaxed text-muted-foreground">
                Disclaimer: This is an indicative estimate using a rule-based model, not a
                guarantee. Final approval depends on the lender's policies.
              </p>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

const inputCls =
  "w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-ring";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      {children}
    </label>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border/60 pb-2 last:border-0">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium">{children}</dd>
    </div>
  );
}
