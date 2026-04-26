namespace Capstone2PipelineViewer;

/// <summary>Preset-aligned sample queries (same lists as the original PipelineViewer).</summary>
public static class TestQueries
{
    public const string WhoMalariaPreset = "who-malaria";
    public const string UgandaPreset = "uganda";

    public static readonly IReadOnlyList<string> WhoMalaria = new[]
    {
        "What is the treatment for uncomplicated Plasmodium falciparum malaria?",
        "Dosing artemisinin-based combination therapy in children under 5",
        "Severe malaria definition and management",
        "When to refer a patient with malaria to hospital?",
        "Pregnancy and malaria treatment recommendations",
        "Drug interactions with artemether lumefantrine",
        "Prophylaxis for travelers to endemic areas",
        "Rapid diagnostic test interpretation false positives",
        "G6PD deficiency and primaquine",
        "Malaria vaccine recommendations RTS,S R21",
        "Resistance to artemisinin in Southeast Asia",
        "Hypoglycemia in severe malaria",
        "Fluid management in severe malaria adults",
        "Exchange transfusion malaria criteria",
        "Cerebral malaria supportive care",
        "Artesunate dose for severe malaria IV",
        "Rectal artesunate pre-referral children",
        "Malaria in HIV coinfection",
        "Species Plasmodium vivax relapse treatment",
        "Monitoring after antimalarial treatment failure",
        "Quality assurance microscopy",
        "Integrated community case management fever",
        "Ethics of placebo-controlled malaria trials",
        "Vector control bed nets IRS",
        "Elimination strategies and surveillance",
    };

    public static readonly IReadOnlyList<string> Uganda = new[]
    {
        "Integrated management of childhood illness pneumonia classification",
        "Diarrhea dehydration ORS zinc treatment plan",
        "HIV antiretroviral therapy first-line regimen adults",
        "Tuberculosis treatment regimen and contact investigation",
        "Malaria uncomplicated case management ACT dosing children",
        "Postpartum hemorrhage emergency management oxytocin",
        "Family planning contraceptive counseling methods",
        "Hypertension diagnosis and management primary care",
        "Diabetes mellitus type 2 glycemic targets",
        "Acute stroke referral and supportive care",
        "Syndromic management sexually transmitted infections",
        "Cervical cancer screening VIA HPV",
        "Routine immunization schedule infants Uganda",
        "Severe acute malnutrition inpatient management",
        "Tuberculosis preventive therapy isoniazid",
        "Depression screening and management primary care",
        "Asthma chronic management inhaler technique",
        "Chronic kidney disease staging referral",
        "Exclusive breastfeeding six months",
        "Pre-eclampsia severe features magnesium sulfate",
        "Sepsis empirical antibiotics adults",
        "Rabies post-exposure prophylaxis dog bite",
        "Burns initial wound care and referral",
        "Snake bite envenomation hospital referral",
        "Neonatal sepsis danger signs referral",
    };

    public static IReadOnlyList<string> ForPreset(string preset) =>
        preset.Equals(UgandaPreset, StringComparison.OrdinalIgnoreCase) ? Uganda : WhoMalaria;
}
