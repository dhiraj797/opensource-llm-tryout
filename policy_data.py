"""
Policy Data Module - Chunked RBIN Expense & Travel Policy Documents
===================================================================
Contains structured policy text chunks for RAG-based retrieval.
Each chunk is a dictionary with 'title', 'section', 'content' and 'keywords'.
"""

POLICY_CHUNKS = [
    # =========================================================================
    # NTRE (Non-Travel Related Expenses) Reimbursement Policy
    # =========================================================================
    {
        "title": "NTRE Reimbursement Policy - Overview",
        "section": "NTRE",
        "content": (
            "Non-Travel Related Expenses (NTRE) Reimbursement Policy applies to M&SS and M&SS Trainees of RBIN. "
            "The tool used is ezyClaim (TMS for M&SS in MA division until further notice). Currency is INR. "
            "Advances are NOT issued for NTRE."
        ),
        "keywords": ["ntre", "non-travel", "ezyclaim", "reimbursement", "overview", "scope"],
    },
    {
        "title": "NTRE - Inclusions",
        "section": "NTRE",
        "content": (
            "NTRE covers the following expense types: "
            "1) Business Entertainment Expenses (not covered in travel guidelines), "
            "2) Company Vehicle Expenses (common pool), "
            "3) Office Maintenance Expenses (exceptional cases), "
            "4) Local Conveyance, "
            "5) Development and Marketing Expenses, "
            "6) Other Miscellaneous Expenses (see Annexure), "
            "7) Credit Card Settlements (refer to GL/R/01 and GL/R/02), "
            "8) Petro/Fuel Card Settlements (refer to GL/P/39)."
        ),
        "keywords": ["ntre", "inclusions", "entertainment", "vehicle", "maintenance", "conveyance", "marketing", "credit card", "fuel card", "claim", "expenses", "covered", "eligible"],
    },
    {
        "title": "NTRE - Exclusions",
        "section": "NTRE",
        "content": (
            "NTRE does NOT cover: "
            "1) Business Travel Related Expenses (covered under GL/R/01 and GL/R/02), "
            "2) Unofficial/Personal Expenses, "
            "3) Payments already paid via salary."
        ),
        "keywords": ["ntre", "exclusions", "not covered", "personal", "travel", "salary", "cannot claim", "not allowed", "unofficial"],
    },
    {
        "title": "NTRE - Approval Matrix",
        "section": "NTRE",
        "content": (
            "NTRE Approval Matrix (in INR): "
            "Group Leader: Up to 21 INR. "
            "Department Head: Up to 420 INR. "
            "LD Members, Two-letter unit code managers (PC, PT, FC, MG, EO, RP, HR and others): Up to 4,200 INR. "
            "GL: Over 4,200 INR. "
            "Reference: GL/P/22 - Bill passing of exceptional transactions."
        ),
        "keywords": ["ntre", "approval", "matrix", "limit", "group leader", "department head", "gl", "amount"],
    },
    {
        "title": "NTRE - Workflow",
        "section": "NTRE",
        "content": (
            "NTRE Claim Workflow: "
            "1) Associate submits claim in ezyClaim. "
            "2) GS/HRS9-IN team verifies claim. "
            "3) Claim routed to approving authority based on matrix. "
            "4) Post-approval, claim processed and reimbursement done. "
            "5) Rejected claims returned to associate. "
            "6) Non-compliant claims not processed. "
            "7) Claims should be submitted within the same quarter."
        ),
        "keywords": ["ntre", "workflow", "process", "submit", "approve", "reject", "quarter", "timeline"],
    },
    {
        "title": "NTRE - Local Conveyance",
        "section": "NTRE",
        "content": (
            "Local Conveyance under NTRE: "
            "Distance Limit: Less than 100 km. "
            "Duration Limit: Less than 8 hours. "
            "Reference: Annexure 2 (Sections 1 & 2) of GL/R/01."
        ),
        "keywords": ["ntre", "local conveyance", "distance", "100 km", "8 hours", "duration"],
    },
    {
        "title": "NTRE - General Principles",
        "section": "NTRE",
        "content": (
            "General Principles for NTRE: "
            "ezyClaim replaces SAATHI. "
            "Prioritize purchasing channels (e.g., My i-Buy) or HR processes. "
            "ezyClaim is for exceptions only, with justification. "
            "Approvers must ensure claims cannot be processed via other channels."
        ),
        "keywords": ["ntre", "ezyclaim", "saathi", "purchasing", "i-buy", "principles", "exceptions"],
    },
    {
        "title": "NTRE - Expense Categories (Annexure 1)",
        "section": "NTRE",
        "content": (
            "NTRE Expense Categories: "
            "Business entertainment expenses: Entertainment with team/guests while not traveling. "
            "Office maintenance expenses: Statutory payments, utilities, one-time repairs. "
            "Company vehicle maintenance expenses: Minor repairs to company pool/demo cars. "
            "Development and Marketing expenses: Competitor samples (up to INR 25K), minor marketing expenses (up to INR 10K). "
            "Local conveyance expenses: Local travel, parking, tolls. "
            "Other miscellaneous expenses: Government payments, stamp papers, notary charges."
        ),
        "keywords": ["ntre", "category", "entertainment", "maintenance", "marketing", "competitor", "samples", "parking", "tolls", "notary", "food", "meals", "team", "birthday", "celebration", "cake", "lunch", "dinner", "party", "business entertainment"],
    },
    {
        "title": "NTRE - Prohibited Actions & Gift Policy",
        "section": "NTRE",
        "content": (
            "NTRE Prohibitions: "
            "Splitting NTRE requests to avoid standard purchase process is prohibited. "
            "Gifts for associates should be routed through SHABASH or Central Purchase. "
            "Gifts for third parties should follow RO-IN gratuity guidelines."
        ),
        "keywords": ["ntre", "prohibited", "splitting", "gifts", "shabash", "gratuity", "birthday", "gift", "present", "reward"],
    },

    # =========================================================================
    # Inland Travel and Local Conveyance Policy (GL/R/01)
    # =========================================================================
    {
        "title": "Inland Travel Policy - Overview & Definitions",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Inland Travel and Local Conveyance Policy (GL/R/01) applies to all RBIN associates traveling within India on company business, covering all RBIN locations. "
            "Inland Travel: Travel more than 100 km one-way from workplace AND trip duration more than 8 hours. "
            "Local Conveyance: Travel less than 100 km one-way from workplace AND trip duration less than 8 hours. "
            "Adheres to Central Directive CD02603-International Mobility."
        ),
        "keywords": ["inland travel", "local conveyance", "definition", "100 km", "8 hours", "gl/r/01", "domestic"],
    },
    {
        "title": "Inland Travel - Associate Categories",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Associate Categories for Travel: "
            "Category A: Blue-collar, apprentices, contract staff. "
            "Category B: M&SS Group VI & VII, Graduate/Trainee Engineers, Commercial Management/Technical Trainee, Management Apprentice. "
            "Category C: M&SS Group IV (including JMPs) and V. "
            "Category D: M&SS SL1 & Group III. "
            "Category E: M&SS SL2. "
            "Category F: VP, SVP, GL. "
            "Business Unit Head can authorize SL1 if no SL2 available."
        ),
        "keywords": ["category", "associate", "grade", "group", "sl1", "sl2", "vp", "svp", "gl", "blue-collar"],
    },
    {
        "title": "Inland Travel - Booking & Accommodation",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Travel Booking: Use ezyTrip Requests (bosch.com). Service provider books and sends voucher. "
            "Accommodation: Use approved service provider for guesthouses, serviced apartments, and hotels. "
            "Prioritize guest houses/contracted properties (no skip level approval). If unavailable, associate can book with prior notice to travel desk and manager. "
            "Adhere to Annexure 1 limits (exception requires sanctioning authority approval). "
            "Domestic air travel: Book at least 14 days in advance (exceptions require approval). "
            "Travel should be economical. Avoid flight changes/cancellations."
        ),
        "keywords": ["booking", "ezytrip", "accommodation", "hotel", "guesthouse", "air travel", "14 days", "advance booking", "flight"],
    },
    {
        "title": "Inland Travel - Personal Trip Combined with Business",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Personal trips can be combined with business trips at no extra cost to company. "
            "Limit personal leave days to official travel days. Manager approval needed. "
            "For non-RBIN associate travel (guests, visitors, etc.) where company bears expenses, SL3 approval needed."
        ),
        "keywords": ["personal trip", "combined", "leave", "guest", "visitor", "sl3"],
    },
    {
        "title": "Inland Travel - Travel Advance",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Travel Advance: Approved by sanctioning authority (Annexure 1). Minimized as much as possible. "
            "Request via ezyTrip Indent. Released by GS/HRS9-IN 7 days before travel based on eligibility (overnight stays and daily allowance). "
            "Up to Rs. 2,000/day extra for local travel during outstation trips (sanctioning authority approval). "
            "Borrowing from customers/suppliers during travel is strictly prohibited."
        ),
        "keywords": ["advance", "ezytrip", "indent", "7 days", "2000", "local travel", "borrowing", "prohibited"],
    },
    {
        "title": "Inland Travel - Corporate Credit Cards",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Corporate Credit Cards follow CD-02381 (Payment Transactions and Electronic Banking). "
            "Issued to SL2+ (Below SL2 requires Department Head proposal and Sanctioning Authority approval, max INR 150,000 limit). "
            "Not for settling other associates' expenses. Can be used for conference/seminar registration (pre-approval required). "
            "Official use only. Fuel excluded. Expenses itemized in travel statement. "
            "No travel advance if holding corporate credit card. Petty cash expenses allowed. "
            "Upload credit card statements to Saathi/TMS within 7 days (MA associates use TMS, others use Saathi). "
            "Late submissions over 30 days: Salary recovery, refunded on statement receipt. "
            "Personal use: Repay electronically within 7 days."
        ),
        "keywords": ["credit card", "corporate", "sl2", "150000", "7 days", "30 days", "salary recovery", "fuel", "conference"],
    },
    {
        "title": "Inland Travel - Outstation Candidate Interviews",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Outstation Candidates for Interview: Interviews preferably via video conference (final round face-to-face allowed). "
            "Reimbursement based on the travel category of the position interviewed for (Annexure 1). "
            "Lodging & boarding: If arriving previous day or staying overnight: Full per diem. "
            "If arriving same day and not staying: 50% daily allowance. "
            "NEFT requisition requires HRL & HRL-M approval."
        ),
        "keywords": ["interview", "candidate", "outstation", "video conference", "per diem", "50%", "daily allowance"],
    },
    {
        "title": "Inland Travel - Allowances (Per Diem, Daily, Lump Sum)",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Allowances: "
            "Per Diem: Full allowance for own lodging arrangements (overnight stay). "
            "Daily Allowance: Full allowance if Bosch arranges lodging, or if staying at company guest house/BLC and covering own boarding expenses. "
            "Full allowance for journey commencement and return days (one-day trip without overnight stay). "
            "Lump Sum Allowance: For residential courses (fee includes boarding/lodging), BLC training stays, or when meals included in hotel/guest house tariff. "
            "50% allowance if lunch/dinner provided by others/company."
        ),
        "keywords": ["allowance", "per diem", "daily allowance", "lump sum", "lodging", "boarding", "50%", "meals"],
    },
    {
        "title": "Inland Travel - Entertainment Expenses",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Entertainment Expenses: As per Annexure 1. Other categories require sanctioning authority approval. "
            "Names/status of those entertained and supporting bills required. "
            "No mutual entertainment within RO-IN. "
            "Lump sum replaces daily allowance for entertainment days."
        ),
        "keywords": ["entertainment", "bills", "mutual", "lump sum", "annexure", "food", "meals", "team", "lunch", "dinner", "guests"],
    },
    {
        "title": "Inland Travel - Conveyance Rules",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Conveyance: Plants/SOs provide cars or hired taxis. Requisitions in advance. "
            "Prioritize prepaid taxis/point-to-point services. Lowest cost, safe options preferred. "
            "Car rentals allowed when practical/economical. One car per group at out-of-town locations. Authorized drivers only. "
            "Use pooled cars where available (business only, submit fuel receipts). "
            "Own vehicle use discouraged (see GL/S/07). Allowed with undertaking, manager approval, insurance, and license. Reimbursement at actuals or per km (Annexure 2)."
        ),
        "keywords": ["conveyance", "taxi", "car rental", "pooled car", "own vehicle", "fuel", "per km", "gl/s/07"],
    },
    {
        "title": "Inland Travel - Settlement of Accounts",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Settlement: Settle hotel bills in full. Obtain invoices with Bosch GST details (use ezyTrip Tool for GST address). "
            "Submit Statement of Travel Expenses via ezyTrip to GS/HRS9-IN within one month along with bills and receipts. Keep originals until settled. "
            "Exceeding limits requires sanctioning authority approval. "
            "Unutilized advance returned electronically. GS/HRS9-IN recovers from next payroll if not returned. "
            "For group bookings (conference, seminar, etc.), GS/PUI2-AP2 approval required. "
            "Associates responsible for timely settlement within stipulated timelines. Claim expenses via ezyTrip only."
        ),
        "keywords": ["settlement", "hotel", "gst", "invoice", "one month", "bills", "receipts", "advance return", "payroll", "group booking"],
    },
    {
        "title": "Inland Travel - Air Travel Restrictions",
        "section": "Inland Travel (GL/R/01)",
        "content": (
            "Air Travel Restrictions: Maximum 2 GLs/executive management (same GB/Corporate dept) on same aircraft/transport. "
            "Avoid excessive associates (same GB/Corporate dept) on same aircraft. "
            "One functional area should use different modes of transport."
        ),
        "keywords": ["air travel", "restriction", "gl", "executive", "aircraft", "safety"],
    },

    # =========================================================================
    # Fuel Card Policy (GL/P/39)
    # =========================================================================
    {
        "title": "Fuel Card Policy - Overview & Eligibility",
        "section": "Fuel Card (GL/P/39)",
        "content": (
            "Fuel Card Policy (GL/P/39): Introduces Fuel Cards for eligible officers (SL2 and above) to improve safety and provide fueling flexibility. "
            "Eliminates fire hazard associated with on-site fuel storage. "
            "Eligibility: SL2 (GM) and above who receive company vehicles and fuel. Card similar to bank credit/debit card (e.g., SBI Fuel Card). "
            "Additional Cards: For VP (SL3) and above, exclusively for second car use."
        ),
        "keywords": ["fuel card", "sl2", "gm", "vp", "sl3", "eligibility", "sbi", "company vehicle", "gl/p/39"],
    },
    {
        "title": "Fuel Card - Credit Limits",
        "section": "Fuel Card (GL/P/39)",
        "content": (
            "Fuel Card Credit Limits (per month, in INR): "
            "GM (SL2): 30,000 (Primary Card only). "
            "Senior GM (SL2): 30,000 (Primary Card only). "
            "Vice President (SL3): 30,000 (Primary Card) and 15,000 (Additional Card)."
        ),
        "keywords": ["fuel card", "credit limit", "30000", "15000", "gm", "vp", "monthly"],
    },
    {
        "title": "Fuel Card - Usage Procedure",
        "section": "Fuel Card (GL/P/39)",
        "content": (
            "Fuel Card Procedure: For fuel purchases at fuel stations only. Consumables (oil, brake fluid, etc.) claimed via reimbursement. "
            "Use at any station accepting credit cards. Normal unleaded fuel only (exceptions for emergencies). "
            "Executives must accompany drivers during fueling. Do not share PIN. "
            "Lost/destroyed cards: Report to issuing bank and BanP/CAR immediately for blocking and replacement. Use reimbursement process until new card arrives. "
            "Return card to Personnel/Sales Head upon separation."
        ),
        "keywords": ["fuel card", "procedure", "fuel station", "pin", "lost card", "separation", "unleaded"],
    },
    {
        "title": "Fuel Card - Settlement & Records",
        "section": "Fuel Card (GL/P/39)",
        "content": (
            "Fuel Card Settlement: Submit monthly bank statement and charge slips/invoices/bills to CAR department by the 10th of each month. "
            "Two consecutive months of non-submission: Card deactivated. "
            "Bills must show vehicle number. Email copies allowed for lost/damaged bills. "
            "Use Petrol/Diesel Consumption Card (Annexure 1, A5 size). Keep card in vehicle at all times for auditing."
        ),
        "keywords": ["fuel card", "settlement", "10th", "monthly", "deactivated", "vehicle number", "consumption card", "audit"],
    },

    # =========================================================================
    # Cheque/DD/NEFT Requisitions (GL/P/22)
    # =========================================================================
    {
        "title": "Bill Passing Policy - Overview",
        "section": "Bill Passing (GL/P/22)",
        "content": (
            "Cheque/DD/NEFT Requisitions and Bill Passing for Exceptional Transactions (GL/P/22): "
            "Defines procedures for cheque, DD, and NEFT requisitions and bill passing for exceptional transactions where Purchase Orders cannot be issued. "
            "Emphasizes electronic transactions to reduce paper-based payments. "
            "Applies to all Business Divisions (GBs) at all RBIN plants, including SOs. "
            "All cheques/DDs must be marked 'Account Payee'."
        ),
        "keywords": ["cheque", "dd", "neft", "bill passing", "exceptional", "purchase order", "gl/p/22"],
    },
    {
        "title": "Bill Passing - Approval Limits",
        "section": "Bill Passing (GL/P/22)",
        "content": (
            "Bill Passing Approval Limits (in INR): "
            "Group Leader: Up to 17.5. "
            "Department Head: Up to 350. "
            "LD Members, Two-letter Designation Officers (PM, FC, MG, EO, RP, NE, HR, etc.): Up to 3,500. "
            "GL: Over 3,500. "
            "Value limits aligned with Central Directive CD-G2 (07.11.2016) with Country Factor for India (50%). "
            "Adding more approvers is not allowed."
        ),
        "keywords": ["bill passing", "approval", "limit", "group leader", "department head", "gl", "cd-g2"],
    },
    {
        "title": "Bill Passing - Procedure",
        "section": "Bill Passing (GL/P/22)",
        "content": (
            "Bill Passing Procedure: "
            "1) Use Cheque/DD Requisition form (5133 XX 0000). "
            "2) Attach the relevant bill/document signed by the approving authority (as per GL/S/03). For multiple items, use a Statement of Expenses. "
            "3) Requisition form signed by the designated signing authority and sent to the relevant department. "
            "4) If signing authority not available, next higher-level authority can sign. "
            "5) Prepared Cheque/DD sent to the person/department specified in the requisition."
        ),
        "keywords": ["bill passing", "procedure", "form", "requisition", "signing authority", "cheque"],
    },
    {
        "title": "Bill Passing - Special Cases",
        "section": "Bill Passing (GL/P/22)",
        "content": (
            "Bill Passing Special Cases: "
            "Maverick Buying (payments without Purchase Orders/Department) is included. "
            "Spares for vehicles/generators/material handling equipment included. "
            "Expenditures for Bosch Fine Arts Society, Bosch Sports Club have specific approval processes (not subject to standard value limits). "
            "BMSI staff/consultant payments handled by RBIN/BMSI, RBIN/HRL, HRC following BMSI guidelines."
        ),
        "keywords": ["maverick", "special case", "spares", "fine arts", "sports club", "bmsi"],
    },

    # =========================================================================
    # Gratuities to Public Officials (CD 03002)
    # =========================================================================
    {
        "title": "Gratuities to Public Officials - Rules",
        "section": "Gratuities - Public Officials",
        "content": (
            "Gratuities to public officials in India are permissible ONLY if ALL criteria are met: "
            "1) No Influence: Must not be given to influence official acts. Gratuities for specific official acts are strictly prohibited. Report any requests to RBIN/CPO. "
            "2) Compliance with official's internal rules and related to duties. "
            "3) Required Approvals: Travel/Hospitality up to EUR 25 and token gratuities (EUR 35 max) require verbal confirmation. "
            "Gratuities exceeding EUR 25 (travel/hospitality) or token value require prior written approval from RBIN/CPO AND the official's competent authority."
        ),
        "keywords": ["gratuity", "public official", "cpo", "eur 25", "eur 35", "approval", "influence", "prohibited"],
    },
    {
        "title": "Gratuities to Public Officials - Token Gifts & Documentation",
        "section": "Gratuities - Public Officials",
        "content": (
            "Token Gifts to Public Officials: Low-value promotional items like pens, keychains, cups. Max EUR 35. "
            "Documentation: All cases including approvals must be documented within the department for 10 years. "
            "CPO Approval must be obtained from RBIN/CPO via email. "
            "Official obtains approval from their competent authority before the gratuity is given. "
            "If prior written approval is impossible (e.g., spontaneous dinner), verbal confirmation acceptable, followed by written confirmation."
        ),
        "keywords": ["token gift", "documentation", "10 years", "cpo", "email", "promotional", "pen", "keychain"],
    },

    # =========================================================================
    # Gratuities in Commercial Dealings (Third Parties)
    # =========================================================================
    {
        "title": "Gratuities - Third Parties Overview",
        "section": "Gratuities - Third Parties",
        "content": (
            "Gratuities in Dealings with Third Parties (Commercial, excluding public officials). Version 3, January 2019. "
            "Gratuities are generally permissible if they meet category requirements. "
            "More stringent regulations of the other company must be observed. "
            "Full documentation and maximum transparency required. Compliance with antitrust, competition, and tax laws is mandatory."
        ),
        "keywords": ["gratuity", "third party", "commercial", "compliance", "transparency", "documentation"],
    },
    {
        "title": "Gratuities - Category I: Operational Use",
        "section": "Gratuities - Third Parties",
        "content": (
            "Category I - Gratuities for Operational Use ('for the company'): "
            "Given to the company. No stipulations or reason to believe no operational use. "
            "Adequate from commercial perspective. Typically used for business purposes (e.g., electric drill for a dealer selling drills). "
            "No significant personal advantage (no luxury goods). Reasonable in frequency and quantity."
        ),
        "keywords": ["gratuity", "category i", "operational", "company use", "business purpose"],
    },
    {
        "title": "Gratuities - Category II: Operational/Private Use",
        "section": "Gratuities - Third Parties",
        "content": (
            "Category II - Gratuities for Operational/Private Use ('for the company or employees'): "
            "Alternative 1 'Passing on over a company': Agreed by contract, warranty from company that gratuity goes to private final consumer, transparent information. "
            "Alternative 2 'Selection by recipient': Contractually agreed with company owner/representative, choice between types with significant portion for company use, company selects, no stipulations, reasonable."
        ),
        "keywords": ["gratuity", "category ii", "operational", "private", "contract", "consumer"],
    },
    {
        "title": "Gratuities - Category III: Promotional Gifts",
        "section": "Gratuities - Third Parties",
        "content": (
            "Category III - Promotional Gifts ('for the consumer'): "
            "From official stock with Bosch logo. Maximum value EUR 35 plus tax per gift (sales price). "
            "No counter-performance expected (not tied to sales targets). No conditions or stipulations. "
            "No regular gratuities (e.g., not one gift per month to same partner). No targeted advertising to customer/supplier employees. "
            "Public transparency is key (e.g., published on website)."
        ),
        "keywords": ["gratuity", "category iii", "promotional", "gift", "eur 35", "bosch logo", "consumer"],
    },
    {
        "title": "Gratuities - Category IV: Events & Category V: Marketing",
        "section": "Gratuities - Third Parties",
        "content": (
            "Category IV - Events: Business context (product presentation, workshop, etc.). Business context at least 70%. "
            "Suitable attendees considered. Restrictions during contract negotiations (reasonable dinner acceptable). "
            "Awards for sales targets require written pre-agreement. Reasonable gratuity. At least one Bosch associate attends. "
            "Category V - Marketing: Similar requirements as events category."
        ),
        "keywords": ["gratuity", "category iv", "category v", "event", "marketing", "70%", "workshop", "sales target"],
    },
    {
        "title": "Gratuities - Category VI: Private Use (Restricted)",
        "section": "Gratuities - Third Parties",
        "content": (
            "Category VI - Other Private Use ('for the employee'): Generally NOT permissible. "
            "Not allowed: non-monetary awards, private travel, cash, vouchers, gifts to spouses, discounts for customer employees. "
            "Exception: Low-value threshold of EUR 35 plus tax per recipient per year (EUR 100 for G/LD/SVP level), excluding sales target awards. "
            "Above threshold allowed in individual cases (acting as company rep, milestone birthdays, etc.) with prior Compliance approval."
        ),
        "keywords": ["gratuity", "category vi", "private", "not permissible", "eur 35", "eur 100", "compliance", "cash", "voucher"],
    },

    # =========================================================================
    # Abroad Travel Policy (GL/R/02)
    # =========================================================================
    {
        "title": "Abroad Travel Policy - Overview & Definitions",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Abroad Travel Policy (GL/R/02) defines rules for RBIN employee travel abroad. "
            "Governs all official international travel. Adheres to CP-CD S02 (Business Travel, January 30, 2017). "
            "Business Travel: Temporary work outside usual facility, less than 3 months. "
            "Short-Term Assignment: More than 3 months, requires HR letter of assignment. "
            "Business Travel to Germany: Must not exceed 30 calendar days."
        ),
        "keywords": ["abroad", "international", "gl/r/02", "3 months", "30 days", "germany", "short-term assignment"],
    },
    {
        "title": "Abroad Travel - Booking & Process",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Travel agent issues e-tickets after MiTrA indent approval. Department heads ensure travel approval. "
            "One week or less to SAARC countries: Range Heads can sanction. Travel must be justified. "
            "If Indian Rupees accepted, use them (per Inland Travel guidelines). "
            "Date change within two months (same calendar year): No re-approval needed if duration/location unchanged. "
            "Video conferencing preferred when available."
        ),
        "keywords": ["abroad", "booking", "mitra", "saarc", "e-ticket", "video conference", "date change"],
    },
    {
        "title": "Abroad Travel - Spouse/Family & Trip Alterations",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Spouse/partner/family: Inform approver beforehand. Associate covers all spouse/family expenses and insurance. "
            "Boarding expenses only (no vouchers). Approver can allow spouse/family for extended business trips. "
            "Trip alterations (extension/shortening): Requires prior approval. "
            "Travel purpose (business or training) must be specified. Number of travel days determines Foreign Currency release. "
            "Submit requests via MiTra."
        ),
        "keywords": ["abroad", "spouse", "family", "extension", "foreign currency", "mitra", "insurance"],
    },
    {
        "title": "Abroad Travel - Air Travel Eligibility & Booking",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Air Travel Eligibility: Up to SL1: Economy. SL2+: Premium Economy. "
            "Business class allowed for flights over 10 hours (non-stop), medical reasons, or urgent travel with no other option (requires GL approval). "
            "GL can revise travel mode based on business conditions. "
            "Bookings: At least 21 days in advance (GL approval for exceptions). Avoid unnecessary connections/layovers/changes. "
            "Internet/Wi-Fi during flight: Requires pre-approval, expensed under telephone."
        ),
        "keywords": ["abroad", "air travel", "economy", "premium economy", "business class", "10 hours", "21 days", "sl1", "sl2", "wifi"],
    },
    {
        "title": "Abroad Travel - Ground Transportation",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Ground Transportation: Lowest cost, safest options preferred. "
            "Car rentals: Allowed if practical/economical, one car per group, authorized drivers only, refuel before return. "
            "Use corporate credit cards, notify rental company and card provider in case of accident. "
            "Pooled cars: Use for business purposes, submit fuel receipts."
        ),
        "keywords": ["abroad", "ground transport", "car rental", "pooled car", "accident", "fuel"],
    },
    {
        "title": "Abroad Travel - Debit Cards & Forex",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Forex released via debit cards according to eligibility and RBI regulations. "
            "Debit card available one day before departure (for trolley, taxi, etc., max $300 advance allowed). "
            "Return unused forex and card immediately upon return. Personal use: Reimburse in foreign currency."
        ),
        "keywords": ["abroad", "debit card", "forex", "rbi", "$300", "advance", "foreign currency"],
    },
    {
        "title": "Abroad Travel - Allowances & VAT",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Allowances: Expenses supported by bills/vouchers. No reimbursement for cigarettes, perfumes, toiletries. "
            "Local conveyance reimbursed with tickets/vouchers. "
            "VAT Refund: RBIN can claim VAT refunds. Bills must contain employee name, Bosch Limited HO address, VAT rate/amount, bill number/date. "
            "Departure/Arrival Allowance: Full allowance if departing before 12:00 or arriving after 12:00; 50% if departing after 12:00 or arriving before 12:00."
        ),
        "keywords": ["abroad", "allowance", "vat", "refund", "bills", "voucher", "12:00", "departure", "arrival", "50%"],
    },
    {
        "title": "Abroad Travel - Out of Pocket & Entertainment",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Out of Pocket Expenses: For seminars/marketing trips with lodging/boarding covered: $50/day with bills or $10/EUR 10 without. "
            "For events with covered lodging/boarding: $10/EUR 10 without bills. "
            "Entertainment Expenses: Forex provided via debit card (pre-approved). Supporting bills required with details of those entertained, date, and purpose. "
            "No mutual entertainment (RO-IN employees)."
        ),
        "keywords": ["abroad", "out of pocket", "$50", "$10", "entertainment", "mutual", "seminar", "marketing"],
    },
    {
        "title": "Abroad Travel - Corporate Credit Cards",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Forex cards/Debit Cards preferred over Indian credit cards due to cost. Local currency corporate credit cards allowed in exceptions. "
            "Corporate credit cards for conference/seminar registration (pre-approval needed). "
            "For eligible employees SL2+, SL1 on need basis. Official use only. "
            "Not for personal or other employee expenses."
        ),
        "keywords": ["abroad", "credit card", "forex card", "conference", "sl2", "sl1"],
    },
    {
        "title": "Abroad Travel - Settlement of Accounts",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Settlement: Submit travel expense statement to Accounts within 15 days of return. Include days/months abroad, exchange granted/returned. "
            "Submit visit report to GM (SL2+) with results achieved. Include approval mail/MiTrA indent and travel statement. "
            "Discrepancies exceeding approved amounts require GM (SL2+) approval regardless of value. "
            "Submit travel statement within 3 months of travel, or advance recovered from salary. "
            "Unspent debit card balance returned/additional expenses settled based on authorized dealer exchange rate."
        ),
        "keywords": ["abroad", "settlement", "15 days", "3 months", "visit report", "gm", "exchange rate", "salary recovery"],
    },
    {
        "title": "Abroad Travel - Air Travel Restrictions",
        "section": "Abroad Travel (GL/R/02)",
        "content": (
            "Air Travel Restrictions for Abroad Travel: "
            "No more than two GLs or two executive management (same GB/corporate department) on same aircraft/transport. "
            "Distribute employees from same GB/Corporate department across different aircraft when possible. "
            "Employees from the same functional area should use different modes of transport."
        ),
        "keywords": ["abroad", "air travel", "restriction", "gl", "executive", "safety", "aircraft"],
    },

    # =========================================================================
    # Policy References (Quick Lookup)
    # =========================================================================
    {
        "title": "Policy References - Quick Lookup",
        "section": "References",
        "content": (
            "Key Policy References: "
            "GL/R/01: Inland Travel and Local Conveyance Policy. "
            "GL/R/02: Abroad Travel Policy. "
            "GL/P/39: Fuel Card Policy. "
            "GL/P/22: Bill Passing of Exceptional Transactions. "
            "GL/S/07: Own Vehicle Use Policy. "
            "CD-02381: Payment Transactions and Electronic Banking. "
            "CD02603: International Mobility. "
            "CP-CD S02: Business Travel (January 30, 2017). "
            "CD-G2: Central Directive for approval limits. "
            "Tools: ezyClaim (NTRE), ezyTrip (Travel), MiTrA (Abroad Travel), Saathi/TMS (Credit Card Statements)."
        ),
        "keywords": ["reference", "gl/r/01", "gl/r/02", "gl/p/39", "gl/p/22", "tool", "ezyclaim", "ezytrip", "mitra", "saathi"],
    },
]
