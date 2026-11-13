/**
 * @file English and Simplified Chinese interface localization. The dictionary handles static labels, while parameterized rules preserve identifiers and measurements in dynamic status messages.
 */

"use strict";

(() => {
  const STORAGE_KEY = "tta_language";
  const requested = new URLSearchParams(location.search).get("lang")
    || document.querySelector('meta[name="tta-lang"]')?.getAttribute("content")
    || "";
  const stored = localStorage.getItem(STORAGE_KEY);
  const locale = /^en(?:-|$)/i.test(requested || stored || "") ? "en" : "zh-CN";
  localStorage.setItem(STORAGE_KEY, locale);

  const EN = {
    "Tap-to-Adapt 在线研究": "Tap-to-Adapt Online Study",
    "报名时请留下邮箱、电话或微信中的至少一项。系统随后会生成匿名内部编号，并依次引导你完成知情同意、调查问卷和报销信息选择。": "Provide at least one email address, telephone number, or WeChat ID when enrolling. The system then generates an anonymous internal ID and guides you through consent, questionnaires, and the reimbursement-information choice.",
    "报名联系方式（至少填写一项）": "Enrollment contact details (provide at least one)",
    "请填写邮箱、电话号码或微信号中的至少一项，用于实验联系与发放报酬。联系方式会单独加密保存，不进入研究分析、模型训练或发布数据集。": "Provide at least one email address, telephone number, or WeChat ID for study contact and compensation. Contact details are encrypted separately and excluded from research analysis, model training, and released datasets.",
    "实验报酬": "Study compensation",
    "完成实验和最终问卷后领取 40 元": "Receive RMB 40 after completing the study and the final questionnaire",
    "完成实验和最终问卷后领取 25 元": "Receive RMB 40 after completing the study and the final questionnaire",
    "加密保存联系方式并开始填写": "Encrypt contact details and begin",
    "实验暂时结束不接收新的受试者，如果您有兴趣请联系study-contact": "The study is temporarily closed and is not accepting new participants. If you are interested, please contact WeChat study-contact",
    "尚未连接": "Not yet connected",
    "编号核验": "Code check",
    "确认": "Confirm",
    "四位尾号": "Four-character ending",
    "请输入四位登录码尾号": "Enter the four-character login code ending",
    "请输入四位登录码尾号。": "Enter the four-character login code ending.",
    "用于核验当前浏览器上的实验编号。": "Used to verify the study code on this browser.",
    "登录码尾号": "Login code ending",
    "重新开始实验（推荐）": "Restart the study (recommended)",
    "从上次中断处继续（会被记录，不推荐）": "Continue from the last interruption (will be recorded, not recommended)",
    "请重新完成网络、播放、点击和语音识别检测后再选择下一步。": "Please redo the network, playback, click, and speech tests before choosing the next step.",
    "去完成检测": "Go to the checks",
    "请重新完成网络检测": "Please redo the network check",
    "请重新完成麦克风检测": "Please redo the microphone check",
    "请重新完成点击检测": "Please redo the click check",
    "请重新完成播放与听辨检测": "Please redo the playback and listening check",
    "未找到匹配的编号，请核对后重试。": "No matching code was found. Check the ending and try again.",
    "已预约受试者入口": "Booked participant entry",
    "若你已经预约实验，请输入四位预约凭证进入。忘记凭证时，填写上方已登记的联系方式或下方姓名，对上后无需凭证也可以继续实验。": "If you already booked a session, enter your 4-digit appointment voucher. If you forgot it, enter the contact details or name you registered; a match lets you continue without the voucher.",
    "姓名（忘记凭证时填写）": "Name (if you forgot the voucher)",
    "四位预约凭证": "4-digit appointment voucher",
    "进入已预约实验": "Enter booked session",
    "请输入四位预约凭证，或填写已登记的联系方式/姓名。": "Enter the 4-digit voucher, or the registered contact details / name.",
    "预约凭证必须是四位数字。": "The appointment voucher must be four digits.",
    "暂时无法进入已预约实验：": "Unable to resume the booked session: ",
    "请妥善保存四位凭证。到达预约时间后输入凭证进入；如凭证遗失，可返回首页用联系方式或姓名进入，也可联系微信 study-contact。": "Keep the four-digit voucher. Enter it after your appointment time. If you lose it, return to the start page and continue with your contact details or name, or contact WeChat study-contact.",
    "报名时必须填写邮箱、电话号码或微信号中的至少一项。": "Enrollment requires at least one email address, telephone number, or WeChat ID.",
    "联系方式已在报名时单独加密保存，无需重复填写。": "Your contact details were encrypted separately during enrollment; you do not need to enter them again.",
    "请在 30 分钟内完成准备并连接手机": "Complete setup and connect the phone within 30 minutes",
    "倒计时从电脑实验会话创建时开始；完成设备检测、手机连接并启动正式实验后自动停止。": "The countdown starts when the computer study session is created and stops after device checks, phone connection, and formal study activation are complete.",
    "开始时间：—": "Started: —",
    "准备时间剩余": "Setup time remaining",
    "准备时限已结束": "Setup time expired",
    "30 分钟准备时间已到": "The 30-minute setup window has ended",
    "此前的电脑实验会话尚未完成准备并启动正式实验，系统已结束该会话并释放实验位置。": "The previous computer session did not finish setup and start the formal study, so the system ended it and released the study slot.",
    "你已返回倒计时开始前的报名界面。已提交的记录会保留，不会被删除；如需继续参与，请重新报名并生成新编号。": "You have returned to the enrollment screen shown before the countdown began. Submitted records are retained and are not deleted. To continue, enroll again and generate a new code.",
    "返回报名并重新开始": "Return to enrollment and restart",
    "准备时间已结束": "Setup time expired",
    "此前的电脑实验会话在 30 分钟内未完成准备并启动正式实验，现已结束。请重新报名并生成新编号。": "The previous computer session did not finish setup and start the formal study within 30 minutes, so it has ended. Enroll again to generate a new code.",
    "Tap-to-Adapt 手机伴随端": "Tap-to-Adapt Companion Phone",
    "在线语音交互研究": "Online voice-interaction study",
    "尚未连接": "Not connected",
    "退出编号": "Sign out",
    "非正式数据模式：当前数据不能并入正式 Study A 或 Study B": "Non-formal data mode: these data cannot enter formal Study A or Study B",
    "输入参与者编号": "Enter your participant code",
    "请在电脑上使用最新版 Chrome 或 Edge。编号由研究团队单独发送；本编号对应一次完整实验。": "Use the latest Chrome or Edge on a computer. The research team provides each code separately; one code corresponds to one complete session.",
    "参与者编号": "Participant code",
    "参与者编号（已从链接填入）": "Participant code (filled from your link)",
    "编号已从专属陪伴链接自动填入，无需手动输入。": "Your code was filled automatically from the companion link. No need to type it.",
    "上方仅为格式示例，请勿输入占位符或自行编写编号。": "The text above is only a format example. Do not enter the placeholder or make up a code.",
    "进入研究": "Enter the study",
    "知情同意": "Informed consent",
    "请完整阅读后作出选择。可选授权不会影响是否能够参加研究。": "Read the entire document before choosing. Optional permissions do not affect eligibility to participate.",
    "请先阅读到文件末尾，再进行电子签署。": "Read to the end of the document before signing electronically.",
    "文件已阅读到末尾，请确认下方授权。": "You have reached the end. Please confirm the permissions below.",
    "电子同意书一式两份：研究团队归档副本写入独立的不可修改同意库；参与者留存副本会在签署后自动下载。两份对应同一记录 ID、文件版本、哈希、签署时间和授权选择。": "Two identical electronic copies are created: the research-team copy is written to a separate immutable consent archive, and the participant copy downloads automatically after signing. Both copies share the same record ID, document version, hash, signing time, and permission choices.",
    "确认并电子签署": "Confirm and sign electronically",
    "补偿登记与手写确认": "Compensation registration and handwritten acknowledgement",
    "此信息只用于核验参与和发放完成实验后的25元补偿，不属于研究数据。": "These details are used only to verify participation and issue the CNY 25 compensation after completion. They are not research data.",
    "此信息只用于核验参与和发放完成实验后的15元补偿，不属于研究数据。": "These details are used only to verify participation and issue the RMB 15 compensation after completion. They are not research data.",
    "请完整阅读本研究知情同意书后勾选同意并打字输入姓名。可选数据授权不会影响是否能够参加研究；手写签名与身份证号仍仅在后续报销页按信任程度自愿提供。": "Please read the study consent form in full, then check agreement and type your name. Optional data permissions do not affect eligibility; handwritten signature and ID number remain voluntary on the later reimbursement page based on trust.",
    "请先打字输入参与者姓名（至少两个字符）。": "Please type the participant name first (at least two characters).",
    "文件已阅读到末尾，请勾选同意、打字输入姓名后确认。": "You have reached the end. Check agreement, type your name, then confirm.",
    "参与者姓名（打字输入）": "Participant name (typed)",
    "学校统一报销信息选择": "University reimbursement information choice",
    "请您提供您的信息，方便我们记录": "Please provide your information so we can keep a record",
    "您可以选择是否提供统一报销所需信息；两种选择均可继续实验。": "You may choose whether to provide the details needed for standard reimbursement. Either choice lets you continue.",
    "请至少留下邮箱、电话号码或微信号中的一项；姓名、证件号与手写签名仅在您选择信任并提供报销身份资料后填写。": "Provide at least one of email, telephone number, or WeChat ID. Your name, identity-document number, and handwritten signature appear only after you choose to trust and provide reimbursement identity details.",
    "身份证号与手写签名仅用于学校统一财务报销；不填写不影响继续实验。请至少留下邮箱、电话号码或微信号中的一项，方便我们联系与发放报酬。": "ID number and handwritten signature are only for university financial reimbursement; skipping them does not affect participation. Please leave at least one of email, phone number, or WeChat ID so we can contact you and issue compensation.",
    "联系方式（至少填写一项）": "Contact details (provide at least one)",
    "请至少留下邮箱、电话号码或微信号中的一项，用于联系、发放实验报酬和处理报销；不会用于研究分析、模型训练或数据集发布。": "Provide at least one of email, telephone number, or WeChat ID for contact, study compensation, and reimbursement. These details are excluded from research analysis, model training, and dataset release.",
    "邮箱": "Email",
    "电话号码": "Telephone number",
    "微信号": "WeChat ID",
    "您此前已完成报销身份资料选择；现在只需补充至少一种联系方式。": "You previously completed the reimbursement identity choice. Now provide at least one contact channel.",
    "加密保存联系方式并继续": "Encrypt and save contact details",
    "未能保存联系方式：": "Could not save contact details: ",
    "请选择是否提供学校报销身份资料": "Choose whether to provide reimbursement identity details",
    "仅在您选择提供并信任后填写": "Complete only if you choose to provide and trust",
    "以下姓名、证件号码和手写签名均为可选报销身份资料；不提供也可以继续实验。": "The name, identity-document number, and handwritten signature below are optional reimbursement identity details. You may continue without providing them.",
    "证件号码和手写签名为可选报销身份资料；姓名沿用知情同意页已填写内容，无需重复填写。不提供也可以继续实验。": "The identity-document number and handwritten signature are optional reimbursement details. Your name is reused from the consent page, so you do not need to type it again. You may continue without providing these details.",
    "姓名将使用知情同意页已填写的姓名。": "Your name will be taken from the consent page.",
    "姓名将使用知情同意页已填写的姓名：{name}": "Your name will be taken from the consent page: {name}",
    "缺少知情同意页姓名，请返回上一页重新签署后再登记报销信息。": "The consent-page name is missing. Return to the previous page, resign consent, then register reimbursement details.",
    "个人调试数据保持隔离": "Personal-debug data remain isolated",
    "请选择一种报酬发放方式": "Choose a compensation route",
    "我愿意提供统一报销所需信息。": "I am willing to provide the details needed for standard reimbursement.",
    "我暂不信任或不愿提供，但仍愿意继续实验。": "I do not currently trust or do not wish to provide them, but I still wish to continue.",
    "不提供身份信息，继续实验": "Continue without providing identity details",
    "个人调试账号禁止填写真实身份信息": "Do not enter real identity information with a personal-debug code",
    "请使用下方按钮填入虚拟调试信息。该记录仍会进入个人调试隔离区，但不能用于领取补偿。": "Use the button below to enter virtual debugging details. The record remains in the personal-debug isolation area and cannot be used to claim compensation.",
    "使用虚拟调试信息": "Use virtual debugging details",
    "为什么需要这些信息": "Why these details are needed",
    "姓名": "Full legal name",
    "证件类型": "Identity-document type",
    "居民身份证": "Mainland resident identity card",
    "护照或其他获准证件": "Passport or another approved document",
    "身份证号／证件号": "Identity-card / document number",
    "系统仅保存加密密文；页面不会在提交后重新显示证件号。": "Only encrypted ciphertext is stored; the page will not display the document number again after submission.",
    "手写电子签名": "Handwritten electronic signature",
    "清除重写": "Clear and rewrite",
    "请用鼠标、触控板或触摸屏在此手写签名": "Use a mouse, trackpad, or touchscreen to handwrite your signature here",
    "请用鼠标、触控板或触摸屏书写。这里保存的是手写确认图样，不宣称构成受信任服务提供者签发的合格电子签名。": "Write with a mouse, trackpad, or touchscreen. This stores a handwritten acknowledgement image; it is not represented as a qualified electronic signature issued by a trust-service provider.",
    "加密保存并继续": "Encrypt, save, and continue",
    "请先在签名框内完成手写电子签名。": "Handwrite your signature in the signature box first.",
    "请至少填写邮箱、电话号码或微信号中的一项联系方式。": "Provide at least one of email, telephone number, or WeChat ID.",
    "请输入有效的邮箱，或改填电话号码/微信号。": "Enter a valid email address, or provide a telephone number or WeChat ID instead.",
    "补偿身份信息说明已更新，请重新阅读并提交。": "The compensation identity notice has changed. Read the new version and submit again.",
    "未能保存补偿登记：": "Could not save compensation registration: ",
    "补偿报销说明已更新，请重新阅读并选择。": "The compensation notice has changed. Read the new version and choose again.",
    "未能保存补偿选择：": "Could not save the compensation choice: ",
    "熟悉流程": "Review procedure",
    "设备检测": "Device checks",
    "连接手机": "Connect phone",
    "点击练习": "Tap practice",
    "开始实验": "Start study",
    "电子同意书 · 一式两份": "Electronic consent · two identical copies",
    "研究团队归档副本已保存": "Research-team archive copy saved",
    "研究服务器已保存本次每一项“同意/未同意”选择、文件版本、哈希和签署时间；参与者副本可随时重新下载并打印。": "The study server saved every yes/no permission, the document version and hash, and the signing time. The participant copy can be downloaded and printed again at any time.",
    "下载参与者留存副本": "Download participant copy",
    "下载可验证证据包": "Download verifiable evidence bundle",
    "先熟悉本次完整实验": "Review the complete session",
    "本次正式交互采用固定的六分钟统计周期，有效交互 30 分钟（5 个活跃块），页面计时流程约 34 分钟且不超过 60 分钟。页面会自动切换对象并提示休息；设备检测和结束问卷不计入正式交互计时。": "The timed interaction uses fixed six-minute measurement blocks: 30 minutes of effective interaction across 5 active blocks, about 34 minutes of page timed flow, and will not exceed 60 minutes. The page automatically switches conversation partners and announces breaks. Device checks and the final questionnaire are outside the timed interaction.",
    "每周期前 4 分钟": "First 4 minutes of each block",
    "和电脑上的实验系统自然对话；说完后等待 1～3 秒，若仍未开始且希望它回答，点击“现在回答”。若已超过约 5 秒仍未自动回答，请重新说一遍，不要再为上一句话点击。": "Talk naturally with the experimental system on the computer. After you finish, wait 1–3 seconds; if it still has not started and you want an answer, tap Answer now. If more than about 5 seconds have passed with no automatic answer, say it again and do not tap for the previous utterance.",
    "随后 2 分钟": "Following 2 minutes",
    "页面提示后改与手机陪伴智能体对话；实验系统若不合适地介入，点击电脑界面中的“停止回答”。": "When prompted, talk with the companion agent on the phone. If the experimental system intervenes inappropriately, press Stop answering on the computer.",
    "周期后休息 1 分钟": "1-minute break after the block",
    "停止对话和点击；休息段不采集音频、不更新模型，也不进入主要统计。": "Stop talking and tapping. No audio is collected, the model is not updated, and the break does not enter primary analysis.",
    "话题本会一直可用": "The topic book is always available",
    "进入正式交互后，页面会显示话题本。它只用于避免冷场，不是准入条件，也不要求按顺序完成；你可以沿着一个话题连续多轮交流，也可以随时自然换题。": "The topic book appears after you enter the timed interaction. It only helps avoid awkward silences; it is not an admission requirement and need not be followed in order. You may continue one topic for several turns or change topics naturally.",
    "设备与环境检测": "Device and environment checks",
    "关闭会议软件，连接电源，保持电脑麦克风位置固定。检测时手机背景声应尚未开始。": "Close meeting software, connect the computer to power, and keep the computer microphone fixed. Phone background audio should not yet be playing.",
    "网络稳定性": "Network stability",
    "等待检测": "Waiting",
    "检测网络": "Test network",
    "正常说话音量与环境噪声": "Normal speaking volume and ambient noise",
    "正常说话音量、环境噪声与语音识别": "Normal speaking volume, ambient noise, and speech recognition",
    "嘴与电脑保持 30–90 厘米；先安静，再按提示自然说一句完整的话。": "Stay 30–90 cm from the computer. Remain quiet first, then say one complete sentence naturally when prompted.",
    "先保持安静，再按提示用正常音量说话": "Stay quiet first, then speak normally when prompted",
    "嘴与电脑、手机均保持 30–90 厘米；先保持安静，再按提示用正常音量说话": "Keep your mouth 30–90 cm from both computer and phone. Stay quiet first, then speak normally when prompted.",
    "测试正常说话音量": "Test normal speaking volume",
    "测试说话音量与识别效果": "Test speaking volume and recognition",
    "请核对识别结果": "Check the recognized text",
    "只要主要意思大致正确即可；少量漏字、近义表达，以及简体与繁体差异都不影响通过。": "The main meaning only needs to be approximately correct. Minor omissions, synonymous wording, and simplified/traditional character differences are acceptable.",
    "大致正确，可以继续": "Approximately correct — continue",
    "明显不对，重新测试": "Clearly incorrect — test again",
    "点击反应延迟": "Tap reaction latency",
    "看到红色圆形后立即点击，共 6 次": "Tap immediately when the red circle appears, for 6 trials",
    "开始反应检测": "Start reaction test",
    "浏览器语音能力": "Browser speech capabilities",
    "浏览器与声音播放检测": "Browser and audio-playback check",
    "手机浏览器与最低音量硬校准": "Phone browser and minimum-volume hard calibration",
    "浏览器与最低音量硬校准": "Browser and minimum-volume hard calibration",
    "播放最低音量校准句": "Play minimum-volume calibration sentence",
    "最低音量校准已通过": "Minimum-volume calibration passed",
    "我已将系统媒体音量调到至少 50%": "I set system media volume to at least 50%",
    "建议 50–70%，关闭静音模式和耳机连接，并按上方说明固定手机。手机网页无法直接读取侧边音量条，调好后请在这里确认。": "Recommended: 50–70%. Disable silent mode and headphones, and fix the phone as instructed above. A phone webpage cannot read the hardware volume slider, so confirm here after adjusting it.",
    "我已将手机系统媒体音量设为至少 50%（建议 50–70%），关闭静音模式和耳机连接，并按说明固定手机。勾选后还必须通过低增益听音校准。": "I set phone system media volume to at least 50% (recommended 50–70%), disabled silent mode and headphones, and fixed the phone in place. I must still pass the low-gain listening calibration.",
    "坐在自然说话位置，让嘴与手机、电脑麦克风均保持 30–90 厘米；按下方说明将手机固定在正前方（肚子到胸口高度），保持充电和屏幕常亮。全部检测（含最低音量硬校准）通过后才能连接。": "Sit naturally, keep your mouth 30–90 cm from the phone and computer microphone, fix the phone in front of you at abdomen-to-chest height, keep it charging, and keep the screen awake. All checks, including minimum-volume hard calibration, must pass before connection.",
    "正在检查": "Checking",
    "重新检查": "Check again",
    "请选择语言并播放测试句": "Choose a language and play the test sentence",
    "测试语音语言": "Test-sentence language",
    "中文": "Chinese",
    "英文": "English",
    "播放测试句": "Play test sentence",
    "请输入你听到的内容": "Type what you heard",
    "请输入听到的句子": "Type the sentence you heard",
    "核对听辨内容": "Check typed answer",
    "正在播放测试句": "Playing the test sentence",
    "测试句已播放，请输入听到的内容": "The sentence has played. Type what you heard.",
    "低增益校准句已播放，请完整输入听到的内容": "The low-gain calibration sentence has played. Type the full sentence you heard.",
    "先把媒体音量设为至少 50%，再播放低增益测试句": "Set media volume to at least 50%, then play the low-gain calibration sentence.",
    "先把系统媒体音量调到 50% 以上，并勾选本卡内确认框": "Set system media volume above 50%, then tick the confirmation box in this card.",
    "请先把系统媒体音量调到 50% 以上，并勾选本卡内确认框": "First set system media volume above 50%, then tick the confirmation box in this card.",
    "已确认系统媒体音量至少为 50%，请播放低增益校准句": "Media volume of at least 50% is confirmed. Play the low-gain calibration sentence.",
    "最低音量硬校准通过，可以连接电脑": "Minimum-volume hard calibration passed. You may connect to the computer.",
    "请先将手机系统媒体音量设为至少 50%，并勾选下方确认框": "Set phone system media volume to at least 50% and tick the confirmation box below.",
    "最低音量校准失败，请提高系统媒体音量后重试": "Minimum-volume calibration failed. Raise system media volume and retry.",
    "请先播放测试句": "Play the test sentence first",
    "听辨正确，浏览器与声音播放正常": "Answer matched. Browser audio playback works.",
    "内容未匹配，请重听后再输入": "The answer did not match. Replay the sentence and try again.",
    "听辨内容未匹配，请重听后完整输入听到的句子": "The heard sentence did not match. Replay it and type the complete sentence.",
    "听辨内容过短，请完整输入听到的句子": "The typed answer is too short. Type the complete sentence you heard.",
    "校准播放增益未按 0.5 记录，不是句子输错。请强制刷新陪伴页后重新播放并核对；若仍失败，请用系统浏览器打开陪伴链接": "Calibration playback gain was not recorded as 0.5. This is not a sentence-mismatch. Force-refresh the companion page, replay, and retry. If it still fails, open the companion link in the system browser.",
    "还没有播放最低音量校准句，请先点击播放": "The minimum-volume calibration sentence has not been played yet. Tap play first.",
    "最低音量硬校准记录不完整，请重新播放校准句": "The minimum-volume calibration record is incomplete. Replay the sentence.",
    "请将系统媒体音量调到至少 50%": "Set system media volume to at least 50%.",
    "请勾选“系统媒体音量已调到至少 50%”": "Tick “system media volume is at least 50%”.",
    "最低音量硬校准未通过，请按卡片提示重试": "Minimum-volume calibration did not pass. Follow the card and retry.",
    "未通过：": "Did not pass: ",
    "测试句播放失败，请检查系统音量后重试": "The sentence could not play. Check system volume and try again.",
    "检测结果未能保存，请重试": "The check result could not be saved. Try again.",
    "缺少：": "Missing: ",
    "请等待": "Please wait",
    "点我": "Tap me",
    "点我（还差 6 次）": "Tap me (6 remaining)",
    "连接手机伴随端": "Connect the companion phone",
    "在手机浏览器打开专属链接": "Open the dedicated link in the phone browser",
    "请现在用手机打开此专属链接": "Open this dedicated link on your phone now",
    "小提示": "Tip",
    "推荐在电脑上登录微信等聊天软件，将复制的链接发送到手机后直接打开；请勿逐字手动输入，以免输错。": "We recommend signing in to WeChat or another messaging app on this computer, sending the copied link to your phone, and opening it there. Do not type the link character by character, as errors are easy to make.",
    "页面会自动带入当前实验编号，无需再次输入。": "The page fills in the current participant code automatically; you do not need to enter it again.",
    "请使用完整编号（TTA-XXXX-XXXX）打开专属陪伴链接，不要只使用四位尾号。": "Open the dedicated companion link with the full code (TTA-XXXX-XXXX). Do not use the four-character ending alone.",
    "完整陪伴链接暂不可用。请使用首次报名的电脑，或联系微信 study-contact 获取完整编号（TTA-XXXX-XXXX），不要使用四位尾号链接。": "The full companion link is not available on this computer. Use the computer that first enrolled, or contact WeChat study-contact for the complete code (TTA-XXXX-XXXX). Do not use a four-character-ending link.",
    "只有两端同时在线且手机端网络、麦克风、点击反应、浏览器与声音播放检测全部通过，本次实验才会启动。请坐在自然说话位置，让嘴与电脑麦克风、手机均保持 30–90 厘米，并固定设备位置。": "The session starts only when both devices are online and the phone network, microphone, tap reaction, browser, and audio-playback checks all pass. Sit naturally, keep your mouth 30–90 cm from the computer microphone and phone, and keep both devices fixed.",
    "等待手机连接": "Waiting for phone",
    "从话题本选择开场话题（可选）": "Choose an opening topic (optional)",
    "正在载入话题本": "Loading topic book",
    "默认可自由交谈；选择一个提示只为帮助开场，交流中仍可自然换题。": "Free conversation is the default. A prompt only helps you begin; you may still change topics naturally.",
    "开始前还需完成以下条件：": "Complete these requirements before starting:",
    "本次实验尚未创建": "This study session has not been created",
    "开始完整实验": "Start the complete session",
    "开始点击练习": "Start tap practice",
    "练习通过，开始完整实验": "Practice passed — start the complete session",
    "设备与手机条件已满足；请先完成点击练习。": "Computer and phone checks are complete. Finish tap practice before starting.",
    "请先在手机上打开上方专属链接，并完成手机端检测。": "Open the dedicated link above on your phone and complete the phone checks first.",
    "请先完成上方仍未通过的检测条件。": "Complete the remaining checks above before starting.",
    "关闭点击练习": "Close tap practice",
    "约 2 分钟 · 不进入正式计时与模型学习": "About 2 minutes · excluded from formal timing and model learning",
    "先练习什么时候点击": "Practise when to tap",
    "界面中的“现在回答 / 停止回答”按钮用于纠正响应时机，不是用来推进页面，也不需要每轮点击。请根据当下真实意图判断：": "The Answer now / Stop answering button corrects response timing; it does not advance the page and is not required every turn. Decide from what you genuinely want at that moment:",
    "刚说完：先等待 1～3 秒": "Just finished: wait 1–3 seconds",
    "说完后先等待 1～3 秒，让电脑实验智能体有机会自行回答；如果等待后仍未开始且你希望它现在回答，再点击“现在回答”。": "After you finish, wait 1–3 seconds to give the computer experimental agent a chance to answer on its own. If it still has not started and you want an answer now, tap Answer now.",
    "等待 1～3 秒后仍未回答：现在回答": "Still no answer after waiting 1–3 seconds: Answer now",
    "你已说完并希望得到回答；等待 1～3 秒仍未开始时，点击一次“现在回答”。": "You have finished and want an answer. If it still has not started after waiting 1–3 seconds, tap Answer now once.",
    "超过约 5 秒仍未回答：重新说一遍": "Still no answer after about 5 seconds: say it again",
    "请重新说一遍，不要再为上一句话点击“现在回答”。": "Say it again, and do not tap Answer now for the previous utterance.",
    "它不应开始或继续回答：停止回答": "It should not start or continue: Stop answering",
    "你还没开口、还没说完、正在和手机陪伴智能体对话，或不希望这次回答继续时，点击一次“停止回答”。": "Tap Stop answering once if you have not started, have not finished, are talking with the phone companion, or do not want this answer to continue.",
    "说完且期待回答：开始回答我": "Finished and expecting an answer: Start answering me",
    "你已说完，并希望电脑上的实验智能体回答；等待 1～3 秒仍未开始回答时，点击一次“开始回答我”。": "You have finished speaking and want the computer experimental agent to answer. If it has not started after waiting 1–3 seconds, tap Start answering me once.",
    "还没开口/没说完，或正和手机对话：停止说话": "Not started/finished, or talking with the phone: Stop talking",
    "你还没开口、还没说完，或正在和手机上的陪伴智能体对话；此时电脑实验智能体开始准备或正在回答，就点击一次“停止说话”。": "If you have not started, have not finished speaking, or are talking with the companion agent on the phone, tap Stop talking once when the computer experimental agent prepares or starts answering.",
    "其他情况：不点击": "All other situations: do not tap",
    "电脑实验智能体正在合适地回答、你不期待回答、或页面正在休息时，都不点击。": "Do not tap when the computer experimental agent is answering appropriately, when you do not expect an answer, or during a break.",
    "练习共 7 题。答错后会解释原因，并锁定 5 秒供你阅读，再在同一题重新选择；通过后才可开始正式实验。": "There are seven practice trials. After a wrong choice, read the explanation for 5 seconds before choosing again on the same trial. Pass all seven before the formal session can begin.",
    "开始 7 题练习": "Start seven practice trials",
    "每道新题及每次错答后均需阅读 5 秒": "Read for 5 seconds before each new question and after every wrong answer",
    "请先审题，{n} 秒后可作答": "Review the question first. You can answer in {n} seconds",
    "现在可以作答": "You can answer now",
    "练习": "Practice",
    "答错后需在本题重试": "Retry this trial after a wrong choice",
    "现在回答": "Answer now",
    "可点击操作": "Clickable action",
    "点击“现在回答”按钮": "Click the “Answer now” button",
    "这时不点击，并继续等待": "Do not tap and keep waiting",
    "这时不点击，然后我再重新说一遍": "Do not tap; I will say it again",
    "请与手机上的陪伴智能体互动": "Please interact with the companion agent on your phone",
    "请转向手机，长按圆形按钮与陪伴伙伴说话；电脑实验系统此时无需点击开始回答": "Turn to your phone and long-press the round button to talk with the companion. You do not need to tap Start answering me on the computer now.",
    "请转向手机与陪伴伙伴说话；电脑实验系统此时无需点击": "Turn to the phone and talk with the companion. Do not tap the computer during this window.",
    "请与手机上的陪伴智能体互动：长按手机圆形按钮说话。": "Please interact with the companion agent on your phone: long-press the round button to speak.",
    "按住说话时，手机将本地识别你的语音并发送给陪伴伙伴。": "While you hold to talk, the phone recognizes your speech locally and sends it to the companion.",
    "此浏览器无法本地识别语音，按住说话时将由电脑端麦克风识别。": "This browser cannot run on-device speech recognition; while you hold to talk, the computer microphone will recognize speech.",
    "按住说话时，由电脑端麦克风采集；浏览器识别失败时由后台语音识别自动接手。": "While you hold to talk, audio is captured by the computer microphone. If browser recognition fails, server-side speech recognition takes over automatically.",
    "当前在{label}内置浏览器中，按住说话将由电脑端识别；失败时再尝试手机识别。": "Opened in the {label} in-app browser: hold-to-talk uses computer recognition first, then phone recognition if that fails.",
    "电脑端识别失败，已切换到手机识别，请继续按住说话。": "Computer recognition failed; switched to phone recognition. Keep holding to speak.",
    "电脑端识别失败，已切换到手机识别，请再按住说话。": "Computer recognition failed; switched to phone recognition. Press and hold again to speak.",
    "电脑端识别失败，且此浏览器不支持手机识别。请确认电脑已连接后重试。": "Computer recognition failed, and this browser cannot run phone recognition. Confirm the computer is connected and retry.",
    "电脑端识别暂时失败，请确认电脑实验页保持打开，再按住说话重试。": "Computer recognition failed temporarily. Keep the computer study page open, then press and hold again to retry.",
    "正在重新连接电脑端识别，连接恢复后请再按住说话。": "Reconnecting to computer recognition. After it restores, press and hold again to speak.",
    "停止回答": "Stop answering",
    "点击“停止回答”按钮": "Click the “Stop answering” button",
    "这时不点击": "Do not tap now",
    "点击练习已通过": "Tap practice passed",
    "正式实验中继续按刚才的规则判断。练习行为不会训练模型，也不会进入正式统计。": "Use the same rules in the formal session. Practice actions do not train the model and do not enter formal analysis.",
    "返回并开始完整实验": "Return and start the complete session",
    "准备开始": "Preparing to start",
    "准备进入第 1 个六分钟统计周期": "Preparing to enter six-minute measurement block 1",
    "话题本": "Topic book",
    "请注意：实验不支持暂停": "The study cannot be paused",
    "确认开始实验": "Confirm start",
    "确定现在开始吗？": "Start now?",
    "请阅读上方说明，倒计时结束后方可开始": "Read the notice above. You can start after the countdown.",
    "倒计时已结束，可以开始": "Countdown finished. You may start now.",
    "确定现在开始": "Start now",
    "进入全屏并开始实验": "Enter fullscreen and start study",
    "正在进入全屏…": "Entering fullscreen…",
    "无法进入全屏，实验不能开始。请允许浏览器全屏后重试。": "Fullscreen could not be entered, so the study cannot start. Allow browser fullscreen and retry.",
    "电脑实验系统未处于全屏，实验不能开始。请重新点击开始并允许全屏。": "The computer study system is not fullscreen. Start again and allow fullscreen.",
    "电脑实验系统已退出全屏，实验不能开始。请重新进入全屏后重试。": "The computer study system exited fullscreen. Re-enter fullscreen and retry.",
    "取消": "Cancel",
    "开始后请保持电脑与手机同时在线、本页始终可见，并预留约 34 分钟页面计时（其中 30 分钟有效交互）。若中途离开、切换页面或连接中断，本次实验无效，需从头重新开始。": "After you start, keep both computer and phone online, keep this page visible, and reserve about 34 minutes of page timed flow (30 minutes of effective interaction). If you leave, switch away, or lose connection, this run is invalid and you must restart from the beginning.",
    "手机端尚未就绪": "The phone companion is not ready yet",
    "输入同一编号还不够：手机必须打开上方专属链接，完成全部检测，并点击启用声音后才会与电脑配对连接。": "Entering the same code is not enough: open the dedicated link above on your phone, finish all phone checks, and tap Enable audio before the devices are paired.",
    "请用手机打开专属陪伴链接（微信内打开也可以）；看到手机页显示“已连接/就绪”后再点开始。": "Open the dedicated companion link on your phone (WeChat is fine); wait until the phone shows Connected/Ready, then start.",
    "查看提示": "View tip",
    "无法开始，需要在手机端打开陪伴系统": "Cannot start: open the companion system on a phone",
    "当前设备不是智能手机。请用手机打开专属陪伴链接；微信内打开也可以，电脑和平板不行。": "This device is not a smartphone. Open the dedicated companion link on a phone; WeChat is fine, but computers and tablets are not.",
    "还不能开始实验": "You cannot start yet",
    "请先完成下列仍未通过的条件，然后再点击开始。": "Finish the unmet requirements below, then try starting again.",
    "同编号登录后，还需完成手机检测并启用声音，两端 WebSocket 同时在线才会放行。": "After logging in with the same code, finish phone checks and enable audio so both WebSockets are online.",
    "请按提示处理后重试；若反复失败，请联系微信 study-contact（微信名：Research team）。": "Follow the tips and retry. If it keeps failing, contact WeChat study-contact (WeChat name: Research team).",
    "可选：同意脱敏数据记录与学术发布": "Optional: consent to de-identified data recording and academic release",
    "了解详情": "Learn more",
    "感谢你愿意为研究贡献去标识化数据。": "Thank you for contributing de-identified data to the research.",
    "此为可选项；不勾选也不影响继续实验。": "This is optional; you can continue the study without checking it.",
    "脱敏数据记录与学术发布": "De-identified data recording and academic release",
    "若你勾选此项，研究团队可在去标识化处理后，将你的交互数据用于学术分析与公开发布。": "If you check this option, the research team may use your interaction data for academic analysis and public release after de-identification.",
    "发布内容不包含可直接识别个人身份的原始语音、逐字转写或完整对话原文；仍会先经过脱敏与审核流程。": "Released materials do not include raw speech, verbatim transcripts, or full dialogue that can directly identify you; de-identification and review still apply.",
    "你可以随时不勾选；不勾选不会影响参加实验、完成实验或领取补偿。": "You may leave this unchecked at any time; doing so does not affect participation, completion, or compensation.",
    "我知道了": "Got it",
    "实验已中断": "Study interrupted",
    "返回准备页重新开始": "Return to setup and restart",
    "本次实验不支持暂停。若发生中断，请返回准备页从头重新开始。": "This study cannot be paused. If it is interrupted, return to setup and restart from the beginning.",
    "提前结束实验": "End study early",
    "等待": "Waiting",
    "系统尚未开始回答": "The system has not started answering",
    "实验系统尚未开始回答": "The experimental system has not started answering",
    "实验系统正在准备回答": "The experimental system is preparing an answer",
    "实验系统正在说话": "The experimental system is speaking",
    "你已说完并希望它回答；等待 1～3 秒仍未开始时点击": "Press after you have finished and want an answer, but it still has not started after waiting 1–3 seconds",
    "你已说完并希望实验系统回答；等待 1～3 秒仍未开始时点击": "Press after you have finished and want the experimental system to answer, but it still has not started after waiting 1–3 seconds",
    "还没开口、没说完，或正与手机陪伴伙伴对话时点击": "Tap if you have not started, have not finished, or are talking with the phone companion",
    "实验系统不该回答或响应时机不合适时点击": "Tap when the experimental system should not answer or the timing is wrong",
    "若暂未开口，手机陪伴伙伴将在 {n} 秒后主动打招呼": "If you stay silent, the phone companion will greet you in {n}s",
    "若暂未开口，我将在 {n} 秒后主动打招呼": "If you stay silent, I will greet you in {n}s",
    "手机陪伴伙伴正在主动打招呼（与电脑实验系统不是同一个）": "The phone companion is greeting you (not the computer experimental system)",
    "我（手机陪伴伙伴）正在主动打招呼": "I (your phone companion) am greeting you now",
    "等待手机陪伴伙伴主动打招呼…": "Waiting for the phone companion to greet you…",
    "电脑 · 实验系统": "Computer · Experimental system",
    "手机 · 陪伴伙伴": "Phone · Companion",
    "与手机陪伴伙伴不是同一个": "Not the same as the phone companion",
    "希望智能体现在回答时，点击屏幕": "Tap the screen when you want the agent to answer now",
    "说完并希望它回答；等待 1～3 秒仍无响应时点击": "After you finish and want an answer, tap if it still has not responded after waiting 1–3 seconds",
    "可以从右侧任选一个话题开始说": "Choose any topic on the right and start speaking",
    "系统正在回答，此处仅显示你的发言": "The system is answering; only your speech appears here",
    "对话话题本": "Conversation topic book",
    "仅用于避免冷场；可连续多轮交流，也可随时自然换题。": "Only to avoid awkward silences. Continue for multiple turns or change topics naturally.",
    "请转向手机继续对话": "Continue the conversation on your phone",
    "请看手机，与陪伴系统对话 · 本轮 2 分钟": "Look at your phone and talk with the companion · 2 minutes",
    "请看手机，与陪伴伙伴对话 · 本轮 2 分钟": "Look at your phone and talk with the companion · 2 minutes",
    "请转向手机，长按圆形按钮与手机对话伙伴交流。电脑端仍在监听：若实验系统不合适地介入，请点击下方「停止说话」。": "Turn to your phone and long-press the round button to talk with the companion. The computer is still listening: if the experimental system intervenes inappropriately, tap Stop talking below.",
    "请转向手机，长按圆形按钮与「手机陪伴伙伴」交流（与电脑实验系统是两个不同个体）。电脑端仍在监听：若实验系统不合适地介入，请点击下方「停止回答」。": "Turn to your phone and long-press the round button to talk with the phone companion (a different agent from the computer experimental system). The computer is still listening: if the experimental system intervenes inappropriately, tap Stop answering below.",
    "现在请看着手机，与陪伴系统对话": "Look at your phone now and talk with the companion",
    "现在请看着手机，与陪伴伙伴对话": "Look at your phone now and talk with the companion",
    "我是手机上的陪伴伙伴，不是电脑上的实验系统。请长按下方圆形按钮对我说话；松开后等待我回答。": "I am the phone companion, not the computer experimental system. Long-press the round button to talk to me; release and wait for my reply.",
    "与实验系统对话 · 本轮 4 分钟": "Talk with the experimental system · 4 minutes",
    "请对着电脑说话。倒计时结束前请保持在此页；若系统接话不合时机，请点「停止说话」。": "Speak to the computer. Stay on this page until the countdown ends; if the system answers at the wrong time, tap Stop talking.",
    "请对着电脑与「实验系统」说话（不是手机上的陪伴伙伴）。说完后等待 1～3 秒；若仍未自动回答且你希望它回答，请点「现在回答」。若已超过约 5 秒仍未回答，请重新说一遍，不要再为上一句话点击。若接话时机不合适，请点「停止回答」。": "Speak to the computer experimental system (not the phone companion). After you finish, wait 1–3 seconds; if it still has not answered automatically and you want an answer, tap Answer now. If more than about 5 seconds have passed with no answer, say it again and do not tap for the previous utterance. If it answers at the wrong time, tap Stop answering.",
    "背景声与对话伙伴": "Background audio and conversation partner",
    "与电脑实验系统是两个不同个体": "A different agent from the computer experimental system",
    "现在与手机对话伙伴交谈": "Talk with the phone conversation partner now",
    "现在与手机陪伴伙伴交谈": "Talk with the phone companion now",
    "我是手机陪伴伙伴，不是电脑实验系统。长按下方圆形按钮对我说话，松开后等待回答。": "I am the phone companion, not the computer experimental system. Long-press the round button to talk to me, then release and wait.",
    "手机对话时段已开始。请长按下方圆形按钮说话。": "The phone conversation window has started. Long-press the round button to speak.",
    "手机陪伴伙伴时段已开始（与电脑实验系统不是同一个）。请长按下方圆形按钮说话。": "The phone companion window has started (not the same as the computer experimental system). Long-press the round button to speak.",
    "请回到电脑端对话": "Return to the computer conversation",
    "请回到电脑端，与实验系统对话": "Return to the computer and talk with the experimental system",
    "不要看我，专心和电脑端互动。": "Do not look at me; focus on the computer.",
    "不要看我（手机陪伴伙伴）。请专心和电脑上的实验系统互动。": "Do not look at me (phone companion). Focus on the computer experimental system.",
    "不要看我（手机陪伴伙伴）。请专心和电脑上的实验系统互动。请不要关闭手机陪伴链接，也不要离开本页。": "Do not look at me (phone companion). Focus on the computer experimental system. Do not close the phone companion link or leave this page.",
    "正在模拟环境声音。请专心和电脑实验系统互动；我是另一个个体。请不要关闭手机陪伴链接。": "Ambient sound is playing. Focus on the computer experimental system; I am a different agent. Do not close the phone companion link.",
    "请不要关闭手机陪伴链接，也不要离开本页。": "Do not close the phone companion link or leave this page.",
    "请不要关闭电脑实验页，也不要离开本页。": "Do not close the computer study page or leave this page.",
    "请不要离开或关闭当前电脑页面。": "Do not leave or close this computer page.",
    "请不要离开或关闭当前手机页面。": "Do not leave or close this phone page.",
    "A. 点击“现在回答”按钮": "A. Click the “Answer now” button",
    "B. 这时不点击": "B. Do not tap now",
    "C. 这时不点击，然后我再重新说一遍": "C. Do not tap now; say it again",
    "正在模拟环境声音，不要看我，专心和电脑端互动。": "Playing ambient sound. Do not look at me; focus on the computer.",
    "正在模拟环境声音。请专心和电脑实验系统互动；我是另一个个体。": "Playing ambient sound. Focus on the computer experimental system; I am a different agent.",
    "实验已暂停": "Study paused",
    "计时和模型更新已经暂停。准备好后继续。": "Timing and model updates are paused. Continue when ready.",
    "继续实验": "Continue study",
    "本段数据采集与模型更新已停止": "Data collection and model updating have stopped for this break",
    "请休息一分钟": "Take a one-minute break",
    "可以放松眼睛和声音，请不要继续对话或点击屏幕。背景声将在休息期间暂停，休息结束后从原位置继续播放。": "Relax your eyes and voice. Do not keep talking or tapping the screen. Background sound pauses during the break and resumes from the same position afterward.",
    "可以放松眼睛和声音，请不要继续对话或点击屏幕。请不要离开或关闭当前手机页面。背景声将在休息期间暂停，休息结束后从原位置继续播放。": "Relax your eyes and voice. Do not keep talking or tapping the screen. Do not leave or close this phone page. Background sound pauses during the break and resumes from the same position afterward.",
    "标准化休息已开始。请休息一分钟，不要继续对话或点击屏幕。": "The standardized break has started. Rest for one minute and do not keep talking or tapping the screen.",
    "标准化休息已开始。请休息一分钟，不要继续对话或点击屏幕，也不要离开或关闭当前手机页面。": "The standardized break has started. Rest for one minute. Do not keep talking or tapping the screen, and do not leave or close this phone page.",
    "每个填空至少填写 20 个汉字，或至少 12 个英文单词。": "Each open-ended answer needs at least 20 Chinese characters or at least 12 English words.",
    "请完整填写所有必答填空题后再提交。": "Please complete all required open-ended answers before submitting.",
    "填空题每个空至少填写 20 个汉字，或至少 12 个英文单词。请补充后再提交。": "Each open-ended answer needs at least 20 Chinese characters or at least 12 English words. Please expand your answers and try again.",
    "请勿切换页面": "Please do not switch away from this page",
    "请勿长时间离开页面（约 30 秒内返回可继续）": "Do not leave this page for long (return within about 30 seconds to continue)",
    "页面暂时离开": "Page temporarily left",
    "电脑实验系统已退出全屏": "The computer study system exited fullscreen",
    "退出全屏已计入离开时间。请在": "Time outside fullscreen counts as away time. Within",
    "检测到实验页不可见。请在": "The study page is not visible. Within",
    "秒内恢复全屏并返回本页，即可继续当前进度。": " seconds, restore fullscreen and return to this page to continue.",
    "仍未进入全屏。请允许浏览器全屏，并在": "Fullscreen is still not active. Allow browser fullscreen within",
    "全程保持电脑实验系统全屏；退出全屏或离开页面共用 30 秒离开时限": "Keep the computer study system fullscreen throughout. Exiting fullscreen and leaving the page share one 30-second away limit.",
    "电脑实验系统退出全屏超过约 30 秒。请从头重新开始。": "The computer study system was outside fullscreen for about 30 seconds. Restart from the beginning.",
    "检测到实验页被切换到后台。请在": "The study page moved to the background. Return within",
    "秒内返回本页，即可继续当前进度。": " seconds to continue from your current progress.",
    "超过时限后需要从头重新开始。": "If the time limit expires, you must restart from the beginning.",
    "我又来了～长按下方圆形按钮继续聊，松开后等待回答。": "I'm back—long-press the round button to keep chatting, then release and wait.",
    "我又来了～我是陪伴系统智能体，咱们再继续聊聊。": "I'm back—your companion agent. Let's keep chatting.",
    "当前同时使用人数较多": "Too many people are using the system right now",
    "为保证语音与网络稳定，请约 {n} 分钟后再来尝试。": "For stable voice and network quality, please try again in about {n} minutes.",
    "请稍后再来": "Please try again later",
    "可以放松眼睛和声音，请不要继续对话或点击屏幕。倒计时结束后，系统会自动进入下一个六分钟统计块。": "Rest your eyes and voice. Do not talk or tap. The next six-minute block begins automatically when the countdown ends.",
    "可以放松眼睛和声音，请不要继续对话或点击屏幕。手机背景声会暂停，倒计时结束后从原位置继续播放。": "Rest your eyes and voice. Do not talk or tap. Phone background audio pauses and resumes from the same position after the countdown.",
    "可以放松眼睛和声音，请不要继续对话或点击屏幕。背景声会暂停，倒计时结束后从原位置继续播放。": "Rest your eyes and voice. Do not talk or tap. Background audio pauses and resumes from the same position after the countdown.",
    "本次实验使用感受": "Experience in this study session",
    "提交": "Submit",
    "问卷及已提交的互动数据已经保存，但本次实验完成确认失败。": "The questionnaire and submitted interaction data were saved, but session completion could not be confirmed.",
    "请保持本页面打开并重试。若持续失败，请联系研究团队并提供错误码": "Keep this page open and retry. If the problem continues, contact the research team and provide error code",
    "重试完成确认": "Retry completion confirmation",
    "本次实验已完成": "Study session complete",
    "计时实验已完成，最终问卷待提交": "Timed session complete; final questionnaire still pending",
    "计时实验数据已保存，但最终问卷尚未提交。请使用首页的受试者凭据继续填写最终问卷后再领取报酬。": "The timed session data are saved, but the final questionnaire has not been submitted. Use the credential on the home page to finish the questionnaire before claiming compensation.",
    "数据已提交。现在可以关闭电脑和手机页面。": "Your data have been submitted. You may now close the computer and phone pages.",
    "最终问卷与实验数据已提交。请按下方说明领取25元实验报酬。": "The final questionnaire and study data have been submitted. Follow the instructions below to claim the CNY 25 compensation.",
    "最终问卷与实验数据已提交。请按下方说明领取15元实验报酬。": "The final questionnaire and study data have been submitted. Follow the instructions below to claim the RMB 15 compensation.",
    "最终问卷与实验数据已提交。请按下方说明领取40元实验报酬。": "The final questionnaire and study data have been submitted. Follow the instructions below to claim the RMB 40 compensation.",
    "计时互动和模型提交已完成，实验位已释放。最终问卷仍须单独提交；计时完成与问卷提交在后台分开记录。请保存此凭据；若现在离开或曾看到完成页，可之后在首页用凭据继续填写。": "The timed interaction and model commit are done, and the experiment slot has been released. The final questionnaire still needs to be submitted separately; timed completion and questionnaire submission are recorded independently. Save this credential. If you leave now or previously saw a completion page, you can continue from the home page later.",
    "已完成的数据已经保存。请联系研究团队确认按正式规则应得的补偿。": "Completed data have been saved. Contact the research team to confirm compensation due under the approved rules.",
    "25元实验报酬": "CNY 25 study compensation",
    "15元实验报酬": "RMB 15 study compensation",
    "40元实验报酬": "RMB 40 study compensation",
    "当前为测试系统：测试数据与正式实验完全隔离，不进入正式分析。": "This is the test system. Its data are fully isolated from the formal study and are excluded from formal analysis.",
    "当前为正式实验入口。开始前请完整阅读知情同意和报销信息说明。": "This is the formal study entrance. Please read the consent and reimbursement notices in full before starting.",
    "暂时无法开始：": "Unable to start: ",
    "如需帮助，请联系微信 study-contact（微信名：Research team）。": "For help, contact WeChat study-contact (WeChat name: Research team).",
    "请联系微信": "Contact WeChat",
    "领取": "to claim it",
    "请发送你的参与者编号尾号": "Send your participant-code suffix",
    "。不要通过微信发送身份证照片；需要的身份信息已经在本页面加密登记。": ". Do not send an identity-card photograph through WeChat; the required identity details were encrypted and registered on this page.",
    "复制微信号 study-contact": "Copy WeChat ID study-contact",
    "复制链接": "Copy link",
    "链接已复制。": "Link copied.",
    "请手动复制上方链接。": "Copy the link above manually.",
    "微信号已复制。": "WeChat ID copied.",
    "请手动复制微信号：study-contact（微信名：Research team）": "Copy the WeChat ID manually: study-contact (WeChat name: Research team)",
    "手机伴随端": "Companion phone",
    "背景声与对话伙伴": "Background audio and conversation partner",
    "未连接": "Not connected",
    "非正式数据模式": "Non-formal data mode",
    "连接到电脑实验": "Connect to the computer study",
    "使用电脑端显示的专属链接时，实验编号会自动带入；手动打开本页时也可以输入同一个编号。本次实验中，两端必须同时在线。": "The dedicated link shown on the computer fills in the participant code automatically. If you open this page manually, enter the same code. Both devices must remain online.",
    "实验会播放手机背景声": "The study will play phone background audio",
    "实验开始前请确保绝对安静": "Ensure absolute quiet before the study",
    "关闭电视、音乐、风扇直吹麦克风，避免他人说话或走动。设备检测与正式交互都需在安静室内进行；若环境噪声偏高，麦克风检测将无法通过。": "Turn off TV, music, and fans blowing on the microphone. Avoid other people talking or moving around. Device checks and the timed interaction must happen in a quiet room; microphone checks will fail if ambient noise is too high.",
    "检测前请确保绝对安静": "Ensure absolute quiet before checks",
    "关闭电视、音乐、风扇直吹麦克风，避免他人说话或走动。安静段与说话段都会计入麦克风检测。": "Turn off TV, music, and fans blowing on the microphone. Avoid other people talking or moving around. Both the quiet and speaking segments count toward the microphone check.",
    "正式交互开始后，手机会持续播放环境背景声（类似咖啡厅或街道环境音），并与对话伙伴语音混合播放。浏览器会以固定增益播放背景声；请将媒体音量设为约 50%，并按下方说明固定手机位置，实验全程勿移动。": "After the timed interaction starts, the phone continuously plays ambient background audio (similar to a café or street scene) mixed with the conversation partner. The browser plays background audio at a fixed gain; set media volume to about 50% and fix the phone as described below without moving it during the session.",
    "请提前知晓：实验会播放背景声": "Please know in advance: the study plays background audio",
    "电脑端开始正式交互后，手机会自动播放连续环境背景声，并在你与手机对话伙伴交流时继续播放。背景声由浏览器按固定增益输出；请现在确认媒体音量约 50%、手机已按说明固定摆放，且未连接耳机。": "After the computer starts the timed interaction, the phone automatically plays continuous ambient background audio and keeps it on while you talk to the phone conversation partner. Background audio is output at a fixed browser gain; confirm media volume is about 50%, the phone is fixed as described, and headphones are disconnected.",
    "继续": "Continue",
    "手机设备与环境检测": "Phone device and environment checks",
    "坐在自然说话位置，让嘴与手机、电脑麦克风均保持 30–90 厘米；按下方说明将手机固定在正前方（肚子到胸口高度），保持充电和屏幕常亮。四项检测全部通过后才能连接。": "Sit in a natural speaking position with your mouth 30–90 cm from both phone and computer microphone. Fix the phone in front of you as described below (between belly and chest height), keep it charging and awake. All four checks must pass before connecting.",
    "手机摆放位置（不用量尺寸）": "Phone placement (no measuring needed)",
    "正前方：": "In front of you: ",
    "手机放在你正前方，屏幕朝向自己。": "Place the phone directly in front of you with the screen facing you.",
    "合适高度：": "Correct height: ",
    "手机底部约在肚子到胸口之间，大约一臂伸直的距离即可。": "The bottom of the phone should be between your belly and chest—about one arm's length away.",
    "固定不动：": "Keep it fixed: ",
    "用支架、书本或手机架固定，实验全程不要移动。": "Use a stand, books, or phone holder to keep it still for the whole session.",
    "浏览器与声音播放": "Browser and audio playback",
    "检测浏览器并播放测试音": "Check browser and play test tone",
    "我听到了测试音": "I heard the test tone",
    "连接前还需完成以下条件：": "Complete these requirements before connecting:",
    "完成检测并连接电脑": "Finish checks and connect to computer",
    "正在连接电脑…": "Connecting to computer…",
    "正在建立配对连接…": "Establishing pair connection…",
    "我已将手机媒体音量设为约 50%，关闭静音模式和耳机连接，并按说明将手机固定在正前方（肚子到胸口高度），嘴与手机、电脑均保持 30–90 厘米。": "I set phone media volume to about 50%, disabled silent mode and headphones, fixed the phone in front of me as described (between belly and chest height), and kept my mouth 30–90 cm from both phone and computer.",
    "手机麦克风音量": "Phone microphone level",
    "先确认距离，再按提示保持安静并用正常音量说话。": "Confirm the distance, then stay quiet and speak normally when prompted.",
    "启用声音并连接": "Enable audio and connect",
    "等待电脑端开始": "Waiting for the computer",
    "背景声将在电脑端开始实验后自动播放。": "Background audio starts automatically after the study begins on the computer.",
    "电脑连接": "Computer connection",
    "背景声": "Background audio",
    "待机": "Standby",
    "实验流程": "Study procedure",
    "本次实验": "This session",
    "播放进度": "Playback progress",
    "进入手机对话时段后，按住下方按钮说话。": "During phone-conversation periods, hold the button below while speaking.",
    "按住说话": "Hold to talk",
    "按住下方圆形按钮说话": "Hold the round button below to talk",
    "长按说话": "Press and hold to speak",
    "长按下方圆形按钮说话": "Press and hold the round button below to speak",
    "手机对话时段开始后可长按圆形按钮说话": "Press and hold the round button once the phone conversation window starts",
    "长按下方圆形按钮说话，松开后等待回答。": "Press and hold the round button below to speak, then release and wait for an answer.",
    "手机对话时段已开始。请长按下方圆形按钮说话。": "The phone conversation window has started. Press and hold the round button below to speak.",
    "正在听，松手发送": "Listening, release to send",
    "正在由电脑麦克风识别：请保持原位、面向手机说话，松手发送": "The computer microphone is recognizing your speech. Stay in place, face the phone, and release to send.",
    "长按说话（请保持原位，面向手机自然说；声音由电脑麦克风识别，无需转向电脑）": "Hold to talk (stay in place and speak naturally toward the phone; the computer microphone will recognize you, so do not turn toward the computer)",
    "按住说话时，语音由电脑端麦克风识别并在手机显示。": "While you hold to talk, speech is recognized by the computer microphone and shown on the phone.",
    "电脑正在识别：": "Computer recognizing: ",
    "我在这儿。想聊的时候，按住下方按钮说话就行。": "I'm here. Hold the button below whenever you want to chat.",
    "请长按下方圆形按钮说话；松开后等待回答。": "Hold the round button below to speak; release and wait for a reply.",
    "现在请看着手机，与陪伴系统对话": "Now look at your phone and talk with the companion",
    "请勿切换应用或锁屏。背景声在对话伙伴回答时也会继续播放；请保持手机位置不变，媒体音量保持 50%。": "Do not switch apps or lock the screen. Background audio continues while the conversation partner answers; keep the phone position unchanged and media volume at 50%.",
    "正在验证": "Verifying",
    "验证失败": "Verification failed",
    "实验进行中": "Study in progress",
    "网络稳定性检测": "Network stability check",
    "麦克风与环境噪声检测": "Microphone and ambient-noise check",
    "点击反应延迟检测": "Tap reaction-latency check",
    "浏览器语音能力检测": "Browser speech-capability check",
    "正在发送 12 次测量请求": "Sending 12 measurement requests",
    "请保持安静 3 秒": "Remain quiet for 3 seconds",
    "请用正常音量自然说一句完整的话，例如介绍今天做过的一件事": "At a normal volume, say one complete sentence naturally, for example something you did today",
    "正在识别刚才的发言": "Recognizing what you just said",
    "请用正常音量说：语音检测一二三": "At a normal volume, say: voice test one two three",
    "正常说话音量偏小": "Normal speaking volume is too low",
    "说话音量过大并出现削波": "Speaking volume is too high and clipping",
    "正常说话音量可用": "Normal speaking volume is suitable",
    "语音识别通过": "Speech recognition passed",
    "语音识别未通过": "Speech recognition did not pass",
    "浏览器能力完整": "Browser capabilities passed",
    "同编号手机已连接，全部检测就绪": "Phone with the same code is connected and all checks are ready",
    "手机已连接，等待完成手机端检测": "Phone connected; waiting for phone checks",
    "手机连接的不是当前实验会话": "The phone is connected to a different session",
    "等待同编号手机连接": "Waiting for a phone with the same code",
    "手机连接已断开": "Phone disconnected",
    "正在重新连接手机伴随端": "Reconnecting the companion phone",
    "所有开始条件均已满足": "All start requirements are satisfied",
    "正在启动实验": "Starting the study",
    "思考中": "Thinking",
    "系统正在准备回答": "The system is preparing an answer",
    "含正在准备回答：你还没开口、没说完，或正在与手机对话时点击": "Includes when it is preparing an answer: press if you have not started, have not finished, or are talking with the phone companion",
    "你还没开口、没说完，或正在与手机对话时点击": "Press if you have not started, have not finished, or are talking with the phone companion",
    "点击可取消这次响应": "Tap to cancel this response",
    "仅当你不希望这次响应继续时点击": "Tap only if you do not want this response to continue",
    "正在回答": "Answering",
    "系统正在说话": "The system is speaking",
    "它不该回答或响应时机不合适时点击": "Press if it should not answer or the response timing is inappropriate",
    "认为响应时机不合适时，点击屏幕": "Tap the screen if the response timing feels inappropriate",
    "仅当它不该回答或时机不合适时点击": "Tap only if it should not answer or the timing is inappropriate",
    "正在生成回复": "Generating a response",
    "无需每轮点击，请按等待时长判断：": "You do not need to tap every turn. Decide by how long you have waited:",
    "说完后等待 1～3 秒；若仍未自动回答且你希望它回答，请点“现在回答”。如果已经超过约 5 秒仍未回答，请重新说一遍，不要再为上一句话点击。": "After you finish, wait 1–3 seconds. If it still has not answered automatically and you want an answer, tap Answer now. If more than about 5 seconds have passed with no answer, say it again and do not tap for the previous utterance.",
    "说完后等待 1～3 秒；仍未回答时点击（超过约 5 秒请重新说一遍）": "After you finish, wait 1–3 seconds and tap if there is still no answer (after about 5 seconds, say it again).",
    "您认为我开始回答您的时机不对时点击": "Tap if you think I started answering at the wrong time.",
    "请保持连续语音对话": "Keep the spoken conversation going",
    "请继续语音互动": "Continue the spoken interaction",
    "互动即将超时": "Interaction about to time out",
    "请留在当前位置，持续与电脑实验系统自然对话；倾听它回答时无需说话。": "Stay in place and keep speaking naturally with the computer experimental system. You do not need to speak while listening to its answer.",
    "请留在当前位置，持续与手机陪伴智能体自然对话；倾听它回答时无需说话。": "Stay in place and keep speaking naturally with the phone companion. You do not need to speak while listening to its answer.",
    "请留在当前位置，持续与当前智能体自然对话；倾听它回答时无需说话。": "Stay in place and keep speaking naturally with the current agent. You do not need to speak while listening to its answer.",
    "现在可以重新作答。": "You can answer again now.",
    "请阅读解释，{n} 秒后可重新选择。": "Read the explanation; you can choose again in {n} seconds.",
    "话题已保存": "Topic saved",
    "自由交谈（需要时再打开话题本）": "Free conversation (open the topic book when needed)",
    "默认自由交谈；需要时可从话题本选择提示": "Free conversation by default; choose a prompt when needed",
    "已恢复保存的开场话题": "Saved opening topic restored",
    "与实验系统对话": "Talk with the experimental system",
    "开场交流": "Opening conversation",
    "自由选题交流": "Free-topic conversation",
    "收尾交流": "Closing conversation",
    "与陪伴系统对话": "Talk with the companion system",
    "标准化休息": "Standardised break",
    "确定提前结束本次实验吗？已完成的数据仍会保存。": "End this study session early? Completed data will still be saved.",
    "实验提前结束": "Study ended early",
    "实验完成": "Study complete",
    "本次实验已提前结束": "Study session ended early",
    "本次实验未完成": "Study session not completed",
    "本次实验已退出": "Study session exited",
    "退出原因": "Reason for exit",
    "本次实验发生中断，需要从头重新开始。": "The study session was interrupted and must be restarted from the beginning.",
    "已完成的过程记录会保留，但不会作为端到端完成数据进入用户实验汇总。": "Completed process records are retained, but they will not enter the user-study summary as an end-to-end completed session.",
    "我知道了，重新准备": "Understood; prepare again",
    "电脑端实验发生中断，需要从头重新开始。": "The computer study session was interrupted and must be restarted from the beginning.",
    "请查看电脑端显示的原因，并在电脑端重新完成准备。": "Check the reason shown on the computer, then complete preparation again on the computer.",
    "我知道了，等待电脑重新开始": "Understood; wait for the computer to restart",
    "实验页面离开或被切换到后台超过 30 秒。": "The study page was left or remained in the background for more than 30 seconds.",
    "电脑实验系统退出全屏超过 30 秒。": "The computer study system remained outside full screen for more than 30 seconds.",
    "计时与模型网络连接意外中断。": "The timing and model network connection was unexpectedly interrupted.",
    "计时模型发生运行错误。": "The timing model encountered a runtime error.",
    "电脑端与手机端未能保持同时在线。": "The computer and phone did not remain online at the same time.",
    "手机陪伴端连接中断且未在允许时间内恢复。": "The companion-phone connection was interrupted and did not recover within the allowed time.",
    "手机陪伴智能体语音生成失败。": "The companion agent failed to generate speech.",
    "电脑实验系统语音生成失败。": "The computer experimental system failed to generate speech.",
    "连续约 30 秒未检测到有效互动，醒目的 30 秒恢复倒计时结束后仍未恢复互动。": "No valid interaction was detected for about 30 seconds, and interaction did not resume during the prominent 30-second recovery countdown.",
    "实验流程被中断。": "The study flow was interrupted.",
    "电脑端与手机端连接中断。": "The connection between the computer and phone was interrupted.",
    "实验流程发生意外中断。": "The study flow was unexpectedly interrupted.",
    "当前网络尚未恢复，请检查网络后重新准备。": "The network has not recovered. Check the connection, then prepare again.",
    "本次实验需要从头重新开始。": "This study session must be restarted from the beginning.",
    "请在电脑端重新完成准备，等待下一次开始。": "Complete preparation again on the computer and wait for the next start.",
    "已连接": "Connected",
    "已配对，正在检测": "Paired; running checks",
    "同编号在线": "Same code online",
    "会话不一致": "Session mismatch",
    "连接断开": "Connection lost",
    "已断开": "Disconnected",
    "已就绪": "Ready",
    "播放中": "Playing",
    "已停止": "Stopped",
    "现在与手机对话伙伴交谈": "Talk with the phone conversation partner now",
    "按住按钮说话，松开后等待回答。": "Hold the button while speaking, then release and wait for an answer.",
    "手机对话时段已开始。": "Phone-conversation period started.",
    "正在模拟环境声音": "Simulating ambient sound",
    "不要看我，专心和电脑端互动。": "Do not look at the phone; focus on the computer interaction.",
    "正在模拟环境声音，不要看我，专心和电脑端互动。": "Ambient sound is playing. Do not look at the phone; focus on the computer.",
    "正在模拟环境声音，不要看我，专心和电脑互动": "Ambient sound is playing. Do not look at the phone; focus on the computer.",
    "电脑实验系统正在插话": "The computer experimental system is intervening",
    "电脑实验系统正在说话": "The computer experimental system is speaking",
    "若接话不合时机，请到电脑点击「停止回答」打断。不要在手机上操作打断。": "If the timing is wrong, tap Stop answering on the computer to interrupt. Do not interrupt from the phone.",
    "请 {time} 后再来和我互动": "Talk with me after {time}",
    "电脑端进入手机对话时段后将自动切换，请稍候": "The phone conversation window will open automatically when the computer reaches that stage. Please wait.",
    "已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页": "This page was replaced by a newer companion page. Close this page and keep only the newest one open.",
    "同编号在线·可开始": "Same code online · ready to start",
    "同编号在线·待确认就绪": "Same code online · waiting for ready",
    "重连中…": "Reconnecting…",
    "回到前台后自动重连": "Will reconnect when you return to this page",
    "同步就绪中": "Syncing readiness",
    "等待电脑端连接": "Waiting for the computer to connect",
    "会话不一致": "Session mismatch",
    "页面关闭": "Page closed",
    "电脑端请关闭此页": "Close this page on the computer",
    "电脑端开始实验后，请先专心与电脑端互动。": "After the computer starts, focus on the computer interaction first.",
    "请回到电脑端，继续与实验系统互动。": "Return to the computer and continue interacting with the experimental system.",
    "语音识别不可用：请允许麦克风，并用系统浏览器（Chrome/Safari）打开，不要使用微信内置浏览器。": "Speech recognition is unavailable. Allow the microphone and open the link in your system browser (Chrome/Safari), not an in-app browser such as WeChat.",
    "正在检查浏览器与语音识别环境…": "Checking browser and speech-recognition environment…",
    "浏览器环境不符合要求": "Browser environment does not meet requirements",
    "当前在{label}内置浏览器中打开，无法使用按住说话功能。请用系统浏览器（Safari/Chrome）打开此链接，不要使用微信内置浏览器。": "This link is open inside the {label} in-app browser, which cannot support hold-to-talk. Open it in your system browser (Safari/Chrome) instead of an in-app browser such as WeChat.",
    "请点击右上角「…」选择「在浏览器中打开」，或复制链接到 Safari / Chrome 后再进入。": "Tap the top-right menu and choose Open in Browser, or copy the link into Safari / Chrome.",
    "应用": "app",
    "正在检查浏览器与语音识别环境": "Checking browser and speech-recognition environment",
    "浏览器与语音识别环境未通过": "Browser and speech-recognition environment check failed",
    "请改用 {browser} 打开": "Open this page in {browser}",
    "当前在{label}内置浏览器中打开，无法参与手机陪伴实验。": "This page is open inside the {label} in-app browser and cannot run the companion study.",
    "麦克风权限被拒绝，请在浏览器设置中允许此站点使用麦克风。": "Microphone permission was denied. Allow this site to use the microphone in browser settings.",
    "无法访问麦克风，请关闭占用麦克风的其他应用后重试。": "The microphone is unavailable. Close other apps using it and retry.",
    "没有检测到语音，请靠近手机并再次按住说话。": "No speech was detected. Move closer to the phone and hold the button again.",
    "当前在{label}内置浏览器中，按住说话将由电脑端识别。": "This link is open inside the {label} in-app browser. Hold-to-talk speech will be recognized on the computer.",
    "本机不支持语音识别，按住说话将由电脑端识别。": "This phone browser cannot run local speech recognition. Hold-to-talk speech will be recognized on the computer.",
    "电脑端识别超时，请确认电脑已连接并重试。": "Computer-side recognition timed out. Confirm the computer is connected and try again.",
    "电脑端未识别到语音。请对着电脑麦克风清晰说话后再试；若仍失败，请改用系统浏览器打开本页。": "The computer did not detect speech. Speak clearly toward the computer microphone and try again; if it still fails, open this page in your system browser.",
    "正在加载中…": "Loading…",
    "正在连接实验系统": "Connecting to the study system",
    "正在连接计时模型与麦克风": "Connecting timing model and microphone",
    "正在进入实验界面": "Entering the study interface",
    "语音识别异常：": "Speech recognition error: ",
    "未知错误": "Unknown error",
    "无法启用麦克风：": "Could not enable the microphone: ",
    "此浏览器不支持语音识别，请改用最新版 Chrome 或 Safari。": "This browser does not support speech recognition. Use the latest Chrome or Safari.",
    "请回到电脑端对话": "Return to the computer conversation",
    "背景声继续播放，请保持手机位置不变。": "Background audio continues. Keep the phone position unchanged.",
    "已暂停": "Paused",
    "正在听，松开结束": "Listening — release to finish",
    "正在使用专属链接自动连接…": "Connecting automatically with your dedicated link…",
    "正在连接…": "Connecting…",
    "正在准备连接…": "Preparing connection…",
    "若未自动进入，请点击继续": "If you are not taken forward automatically, tap Continue",
    "自动连接准备失败，请点击继续": "Automatic connection setup failed. Tap Continue",
    "请输入参与者编号": "Enter your participant code"
    ,"仅用于个人调试：数据和音频与用户实验正式数据永久隔离，不进入用户实验正式统计": "Personal debugging only: data and audio are permanently isolated from formal user-study data and excluded from formal user-study analysis"
    ,"预约未来时间": "Book a future time"
    ,"现在有空位时可直接开始；也可以先预约未来 12 小时内的时间，到点后再用四位凭证进入。": "If a slot is free you can start now, or book a time in the next 12 hours and enter later with the four-digit code."
    ,"当前实验位置空闲。你可以预约未来 12 小时内的时间，到点后再来；每个已预约开始时间的前后 1 小时内均不可再预约。": "The study position is free. You can book a time in the next 12 hours and return later. Times within 1 hour of another booking cannot be selected."
    ,"取消，改为现在开始": "Cancel and start now"
    ,"智能体互动用户实验预约": "Agent-interaction user study appointment"
    ,"为保证实验过程顺利进行，同一预约时段仅安排一位参与者。": "To keep the study running smoothly, only one participant is scheduled in each appointment period."
    ,"预约等待期间本页面不可操作。系统每 15 秒自动检查一次；到达预约时间且实验位置空闲后，会自动开放实验流程。": "This page is locked while you wait. The system checks every 15 seconds and automatically opens the study when your appointment time arrives and the study position is available."
    ,"智能体互动用户实验 · 预约": "Agent-interaction user study · Appointment"
    ,"预约凭证": "Appointment code"
    ,"已发放": "Issued"
    ,"请妥善保存四位凭证。到达预约时间后输入凭证进入；如凭证遗失，请联系微信 study-contact 获取。": "Keep this four-digit appointment code. Enter it when your appointment time arrives; if it is lost, contact WeChat study-contact."
    ,"请输入四位预约凭证": "Enter the four-digit appointment code"
    ,"四位数字": "Four digits"
    ,"验证凭证并进入": "Verify code and enter"
    ,"请选择实验开始时间块（北京时间）": "Choose a study start time block (Beijing time)"
    ,"可预约时间块": "Available appointment time blocks"
    ,"时间块图例": "Time-block legend"
    ,"可选择": "Available"
    ,"已预约或缓冲时间": "Booked or buffer time"
    ,"智能体互动用户实验 · 预约成功": "Agent-interaction user study · Appointment confirmed"
    ,"预约成功": "Appointment confirmed"
    ,"四位预约凭证": "Four-digit appointment code"
    ,"请截图或记下凭证。到达预约时间后输入凭证进入；如凭证遗失，请联系微信 study-contact 获取。": "Take a screenshot or write down the code. Enter it when your appointment time arrives; if it is lost, contact WeChat study-contact."
    ,"复制凭证": "Copy code"
    ,"我已保存，进入等待页": "I saved it — go to the waiting page"
    ,"仅用于个人调试：数据与正式实验永久隔离": "Personal debugging only: data are permanently isolated from the formal study"
    ,"该编号绑定的是已停用的旧版多日流程，不能继续。请在管理端重新生成“仅用于个人调试”或当前实验阶段的新编号。": "This code uses a retired multi-day protocol. Generate a new personal-debugging or current-stage code in the admin console."
    ,"该编号绑定的是已停用的旧版多日流程，请在电脑端换用当前版本的新编号": "This code uses a retired multi-day protocol. Use a new current-protocol code on the computer."
    ,"该编号绑定旧版多日流程，请重新生成当前版本编号": "This code uses a retired multi-day protocol; generate a current-protocol code."
    ,"话题本默认状态尚未保存": "The default topic-book state has not been saved"
    ,"计时模型发生运行错误，数据采集已暂停。请稍后重试；若再次出现，请联系研究团队。": "The timing model encountered an error and data collection is paused. Retry shortly; contact the research team if it happens again."
    ,"计时与模型连接意外中断，数据采集已暂停。点击继续实验将尝试重新连接。": "The timing/model connection closed unexpectedly and data collection is paused. Continue to reconnect."
    ,"服务器检测到电脑端与手机端不再同时在线。请恢复手机连接后继续。": "The server detected that computer and phone are no longer both online. Restore the phone connection before continuing."
    ,"同一编号的手机伴随端已断开。请恢复手机连接后继续。": "The companion phone with this code disconnected. Restore it before continuing."
    ,"手机陪伴智能体语音生成失败，数据采集已暂停。请稍后继续。": "Companion-agent speech generation failed and data collection is paused. Try again shortly."
    ,"同一编号的电脑端与手机端不再同时在线，数据采集已暂停。请恢复手机连接后继续。": "The computer and phone with this code are no longer both online. Data collection is paused; restore the phone before continuing."
    ,"你已暂停实验。计时和模型更新已停止，准备好后继续。": "You paused the study. Timing and model updates have stopped; continue when ready."
    ,"实验页面切换到了后台，数据采集已暂停。请保持本页可见后继续。": "The study page moved to the background and data collection is paused. Keep this page visible before continuing."
    ,"仍在等待同编号的手机伴随端与声音系统恢复。": "Still waiting for the companion phone and audio system with this code to recover."
    ,"正在重新连接计时与模型服务，请稍候。": "Reconnecting the timing and model service. Please wait."
    ,"计时与模型服务尚未恢复，请稍后重试；持续失败时请联系研究团队。": "The timing and model service has not recovered. Retry shortly; contact the research team if the problem continues."
    ,"请完成当前评分触发的补充问题后再提交。": "Complete the follow-up question triggered by your current rating before submitting."
    ,"上次检测未通过，请重新检测": "The previous check did not pass; run it again"
    ,"已通过（从本次记录恢复）": "Passed (restored from this session)"
    ,"尚未完成": "Not completed"
    ,"未通过": "Did not pass"
    ,"同编号的手机伴随端尚未连接": "The companion phone with this code is not connected"
    ,"手机伴随端声音尚未就绪": "Companion-phone audio is not ready"
    ,"手机端检测尚未全部通过并完成连接": "The phone checks have not all passed and connected yet"
    ,"手机网络稳定性检测": "Phone network stability check"
    ,"手机麦克风与环境噪声检测": "Phone microphone and ambient-noise check"
    ,"手机触摸反应延迟检测": "Phone touch reaction-latency check"
    ,"手机浏览器与声音播放检测": "Phone browser and audio-playback check"
    ,"尚未确认手机 50% 媒体音量与固定摆放位置": "Phone media volume at 50% and fixed placement have not been confirmed"
    ,"尚未确认手机媒体音量至少为 50% 与固定摆放位置": "Phone media volume of at least 50% and fixed placement have not been confirmed"
    ,"手机端全部检测已通过，可以连接电脑": "All phone checks passed; you can connect to the computer"
    ,"正在播放测试音": "Playing the test tone"
    ,"若听到了测试音，请勾选下方确认": "If you heard the test tone, confirm below"
    ,"浏览器能力完整 · 测试音已确认": "Browser capabilities passed · test tone confirmed"
    ,"继续自由选题并按真实感受自然对话": "Continue a natural free-topic conversation and act according to how it genuinely feels"
    ,"请根据当前对话自然地简短回应。": "Respond briefly and naturally to the current conversation."
    ,"正在重新确认本次实验完成状态，请保持页面打开。": "Reconfirming session completion. Keep this page open."
    ,"保持手机位置不变，媒体音量保持 50% 不变。": "Keep the phone position unchanged and media volume at 50%."
    ,"此浏览器不支持语音识别，请改用最新版 Chrome。": "This browser does not support speech recognition. Use the latest Chrome."
    ,"环境噪声偏高，请安静后重试。": "Ambient noise is too high. Move to a quieter setting and retry."
    ,"请保持安静 2 秒": "Remain quiet for 2 seconds"
    ,"请保持 30–90 厘米距离，用正常音量连续说：手机语音检测一二三": "Keep a distance of 30–90 cm and speak continuously at a normal volume: phone voice test one two three"
    ,"正常说话音量偏小。请保持 30–90 厘米距离并重试。": "Normal speaking volume is too low. Keep a distance of 30–90 cm and retry."
    ,"说话音量过大并出现削波，请降低音量后重试。": "Speaking volume is too high and clipping. Lower your volume and retry."
    ,"请先在电脑端完成知情同意": "Complete informed consent on the computer first"
    ,"请先在电脑端创建本次实验，再连接手机伴随端": "Create the session on the computer before connecting the companion phone"
    ,"背景声文件无法加载": "The background-audio file could not be loaded"
    ,"背景声格式不受支持，请改用系统浏览器打开": "This browser does not support the background-audio format. Open the link in your system browser instead."
    ,"背景声下载超时，请检查手机网络": "Background-audio download timed out. Check the phone network."
    ,"浏览器阻止了声音播放，请再次长按启用": "The browser blocked audio playback. Press and hold enable again."
    ,"没有找到今天的背景声轨道": "No background-audio track was found for this session"
    ,"手机音频系统没有响应，请关闭其他音频应用后重试": "The phone audio system did not respond. Close other audio apps and retry."
    ,"设备检查 / Device check": "Device check"
    ,"正在检查主实验设备": "Checking the main study device"
    ,"实验系统必须在笔记本或台式电脑浏览器中打开。": "Open the study system in a laptop or desktop browser."
    ,"请在电脑上重新打开": "Reopen it on a computer"
    ,"或": "or"
    ,"如需帮助，请联系微信": "For help, contact WeChat"
    ,"（微信名：Research team）。": "(WeChat name: Research team)."
    ,"为保证语音与网络稳定，请约 30 分钟后再来尝试。": "To keep speech and network stable, please try again in about 30 minutes."
    ,"遇到任何问题，请随时联系微信": "If anything is unclear, contact WeChat"
    ,"（微信名：Research team），我们会及时帮你解答。": "(WeChat name: Research team). We will help promptly."
    ,"开始 Tap-to-Adapt 实验": "Start the Tap-to-Adapt study"
    ,"领取方式": "How to receive it"
    ,"添加微信 study-contact（微信名：Research team），并发送完成页显示的编号尾号": "Add WeChat study-contact (WeChat name: Research team) and send the code ending shown on the completion page"
    ,"遇到问题": "If you have a problem"
    ,"随时联系微信 study-contact（微信名：Research team），我们会及时帮你解答": "Contact WeChat study-contact (WeChat name: Research team) at any time"
    ,"继续填写最终问卷": "Continue the final questionnaire"
    ,"如果你已完成计时互动但尚未提交最终问卷，请输入服务器发放的 8 位受试者凭据。": "If you finished the timed interaction but have not submitted the final questionnaire, enter the 8-character credential issued by the server."
    ,"继续最终问卷": "Continue final questionnaire"
    ,"已锁定实验位置": "Study slot reserved"
    ,"请在预约时间回到本页面。预约时间到达后 10 分钟内输入凭证进入；若上一位尚未结束，请等到其最长时限后再输入。空闲后超过 15 分钟未进入，凭证失效。": "Return to this page at your appointment time. Enter the code within 10 minutes after that time. If the previous session is still running, wait until its limit ends. The code expires if you do not enter within 15 minutes after the slot becomes free."
    ,"预约等待期间请保持本页打开。需要离开可点右上角「退出编号」。系统每 15 秒自动检查一次；上一位结束后请在 15 分钟内输入凭证进入。": "Keep this page open while you wait. Use Sign out at the top right if you must leave. The system checks every 15 seconds; enter the code within 15 minutes after the previous session ends."
    ,"请完整阅读本研究知情同意书后勾选同意并打字输入姓名。可选数据授权不会影响是否能够参加研究；手写签名与身份证号仍仅在后续报销页按信任程度自愿提供。": "Please read the study consent form in full, then check agreement and type your name. Optional data permissions do not affect eligibility; handwritten signature and ID number remain voluntary on the later reimbursement page based on trust."
    ,"电子同意书一式两份：研究团队归档副本写入独立的不可修改同意库；参与者留存副本会在签署后自动下载。两份对应同一记录 ID、文件版本、哈希、签署时间、打字姓名和授权选择。": "Two identical electronic copies are created: the research-team copy is written to a separate immutable consent archive, and the participant copy downloads automatically after signing. Both copies share the same record ID, document version, hash, signing time, typed name, and permission choices."
    ,"请输入您的真实姓名。打字姓名与屏幕勾选确认共同构成电子知情同意记录。": "Please type your real name. The typed name and on-screen acknowledgment together constitute the electronic consent record."
    ,"例如：操作员走查第 1 批": "Example: operator review batch 1"
    ,"例如：操作员走查第 1 批": "Example: operator review batch 1"
    ,"以所选实验类型创建 1 个操作员自测账号；数据、音频和 personal head 使用独立调试目录，不影响本实例的常规实验数据与模型。": "Create one operator self-test account for the selected study type. Data, audio, and the personal head use an isolated debug directory and do not affect this instance's ordinary study data or models."
    ,"以所选实验类型创建 1 个操作员自测账号；数据、音频和 personal head 使用独立调试目录，不影响本实例的常规实验数据与模型。": "Create one operator self-test account for the selected study type. Data, audio, and the personal head use an isolated debug directory and do not affect this instance's ordinary study data or models."
    ,"研究服务器已保存本次每一项“同意/未同意”选择、文件版本、哈希和签署时间；参与者副本可随时重新下载并打印。": "The study server saved every yes/no permission, the document version and hash, and the signing time. The participant copy can be downloaded and printed again at any time."
    ,"Tap-to-Adapt 数据管理": "Tap-to-Adapt data management"
    ,"实验数据管理": "Study data management"
    ,"受试者记录与数据集管理": "Participant records and dataset management"
    ,"未认证": "Not signed in"
    ,"管理员认证": "Operator authentication"
    ,"令牌只保存在当前浏览器标签页，不会写入本地存储。": "The token stays in this browser tab and is not written to local storage."
    ,"管理员令牌": "Operator token"
    ,"进入管理页": "Open the dashboard"
    ,"实验概览": "Study overview"
    ,"管理员令牌下显示受试者姓名与联系方式，并按五个计时窗口列出点击 / 打断次数。论文核心表仍按实验臂聚合，不含身份明文。": "With the operator token, names and contact details are shown, plus tap/interrupt counts for five timing windows. Paper-facing tables still aggregate by condition and omit identity plaintext."
    ,"刷新": "Refresh"
    ,"下载用户实验数据包": "Download the user-study data package"
    ,"用户试验核心": "User-study core"
    ,"其他": "Other"
    ,"数据范围": "Data scope"
    ,"全部 Run": "All runs"
    ,"正在试验中的受试者": "Participants currently in session"
    ,"只读 · —": "Read-only · —"
    ,"刷新看板": "Refresh board"
    ,"只显示已经到场、正在准备或已正式计时的受试者，不含未到场预约。正式开始时间以电脑端计时启动为准。此看板只读取状态，不会改写受试者数据、模型或全局编排。": "Shows only participants who have arrived, are preparing, or are in timed interaction. Booked-but-absent people are omitted. Formal start time follows computer timing start. This board is read-only."
    ,"SHARE FIT 逐人变化趋势": "Share-fit per-person trend"
    ,"share fit组逐人变化趋势看板": "Share-fit per-person trend board"
    ,"只读分析：按模型链顺序看每位 donor 的五块 taps/min 与块内 q（合格纠错 / 机会），IQR 1.5 剔除离群后再算 Spearman 趋势、双侧 p 与是否显著。问卷 11 维已反向计分，越高越好。拟合点预估使用最新开场 q 三人中位闸门，不改写受试者、模型或编排状态。": "Read-only analysis: donor-by-donor five-block taps/min and in-block q (eligible corrections / opportunities), IQR 1.5 outlier removal, then Spearman trend, two-sided p, and significance. The 11 questionnaire dimensions are reverse-scored so higher is better. Fit-point forecasts use the latest opening-q three-person median gate and do not rewrite participant, model, or orchestration state."
    ,"论文核心数据，一屏读取": "Paper-facing core data on one screen"
    ,"参与者均值纠错率": "Mean participant correction rate"
    ,"完成 / 本实例账号": "Completed / accounts on this instance"
    ,"5 blocks / 结束问卷": "5 blocks / exit questionnaire"
    ,"样本进度与主要结局": "Sample progress and primary outcomes"
    ,"统计单位：参与者": "Unit of analysis: participant"
    ,"主要结局只纳入端到端完成实验的参与者；休息、误触和不满足 eligible 条件的 tap 不进入分子或分母。": "Primary outcomes include only end-to-end completers. Breaks, accidental taps, and ineligible taps enter neither numerator nor denominator."
    ,"每位受试者的五窗口点击 / 打断": "Five-window tap / interrupt counts per participant"
    ,"每个窗口显示「点击 / 打断」。点击是电脑端主动接话，打断是停止当前回答。未进入该窗口的格子为 0。身份与口音来自开始前问卷；尚未填写的显示为 —。": "Each window shows tap / interrupt. A tap requests an answer on the computer; an interrupt stops the current reply. Unused windows are 0. Identity and accent come from the pre-study questionnaire; unread fields show —."
    ,"五个六分钟块的组间趋势": "Between-group trend across five six-minute blocks"
    ,"用于 H3/H4 的 condition × block 检查；每格显示参与者均值纠错率、机会数、自动响应暴露量与误触。": "For condition × block checks. Each cell shows mean correction rate, opportunities, automatic-response exposure, and accidental taps."
    ,"可预测性、掌控感与可纠错性": "Predictability, control, and correctability"
    ,"1–7 分；反向题已反向计分，越高表示体验越好。向好性与整体性分开展示，不把同一构念的两类题合成一个均值。不展示开放题原文。": "1–7 scores; reverse items are already reversed so higher is better. Trend and overall items stay separate. Open-text answers are not shown."
    ,"问卷 · 向好性": "Questionnaire · trend"
    ,"问卷 · 整体性": "Questionnaire · overall"
    ,"问卷 · 构念题": "Questionnaire · construct items"
    ,"三臂 × 五块与问卷维度差异": "Three-arm × five-block and questionnaire differences"
    ,"只读计算：仅纳入端到端完成的 ①冷启动在线学 / ②Share 启动在线学 / ③Share 启动冻结。五块拆成上下两张表：上表是各窗口参与者平均纠错率，下表是各窗口人均激活 / 打断次数（行为纠错与实际 tap）。行为纠错包含「本可训练、但冻结臂按设计不更新」的 tap，避免 ③ 被训练口径整列打成 0。问卷拆成向好性 / 整体性 / 构念题三表，不把 trend 与 overall 合成同一维度；1–7 分（反向题已反向），不是百分比。总检验 Kruskal-Wallis，成对 Mann-Whitney + Holm。n 不足时显示「n不足」。不改写任何原始数据、状态或模型。* p<0.05，** p<0.01，*** p<0.001，ns 不显著。": "Read-only: end-to-end completers in (1) cold-start learning, (2) share-init learning, and (3) share-init frozen. The five blocks are two tables: mean correction rate, then mean activate/interrupt counts. Behavioral corrections include taps that could have trained but stay frozen by design, so arm 3 is not forced to 0. Questionnaires stay in trend / overall / construct tables. Kruskal-Wallis plus pairwise Mann-Whitney with Holm. Sparse n shows “n too small”. Nothing is rewritten. * p<0.05, ** p<0.01, *** p<0.001, ns not significant."
    ,"五个六分钟块 · 平均纠错率": "Five six-minute blocks · mean correction rate"
    ,"五个六分钟块 · 平均激活 / 打断次数": "Five six-minute blocks · mean activate / interrupt counts"
    ,"最终问卷凭据与提交状态": "Final-questionnaire credential and submission status"
    ,"计时互动结束并安全提交模型后发放凭据。端到端完成只表示计时互动已完成，不要求最终问卷；「最终问卷已提交」才表示问卷答案已写入。凭据可在端到端完成后继续补填未提交的问卷。上方用户实验汇总按端到端（计时）纳入。": "The credential is issued after timed interaction ends and the model is saved. End-to-end completion means the timed interaction finished, not that the final questionnaire was submitted. The credential can still be used to finish an unsubmitted questionnaire."
    ,"中断后继续 / 重开记录": "Resume / restart records"
    ,"只读事件汇总：记录「从中断处继续」与「重新开始」的次数和时间戳，不改写原始实验数据。": "Read-only event summary of Continue from interruption and Restart counts and timestamps. Original study data are not rewritten."
    ,"公开报名编排": "Public-enrollment orchestration"
    ,"自动进行": "Automatic"
    ,"当前正式实例已从三臂退回 Share 拟合，从链尖 M(16) 继续累计。 自动锁定：改善界 0.2 个百分点、取最低开场平台起点、再确认一个不重叠三人窗；回弹需连续两窗且发生在最低点后再一个三人窗之后；确认帽 22 人。 锁定拟合权重后才会进入 ①冷启动在线学 → ②share 启动在线学 → ③share 启动冻结（每组端到端完成上限 32）。 停止后新受试者暂时无法报名；任意状态下均可查看与导出数据。受试者原始会话与事件不被此页改写。": "This formal instance has returned from the three-arm contrast to share-fit and continues from chain tip M(16). Automatic lock: 0.2 percentage-point improvement bound, lowest opening-plateau start, then one non-overlapping three-person window; rebound needs two consecutive windows after the next window past the minimum; confirmation cap 22 people. Locked fit weights precede (1) cold-start learning → (2) share-init learning → (3) share-init frozen (32 completers each). Stopping blocks new enrollment; viewing and export remain available. Raw sessions are not rewritten here."
    ,"报名状态": "Enrollment status"
    ,"当前阶段": "Current stage"
    ,"当前拟合点": "Current fit point"
    ,"Share 模型链 / 端到端 / 50上限": "Share model chain / end-to-end / cap 50"
    ,"模型就绪待链提交": "Model ready, waiting for chain commit"
    ,"已验证模型最终化": "Verified model finalized"
    ,"① 冷启动学（完成/上限）": "(1) Cold-start learning (done/cap)"
    ,"② share+学（完成/上限）": "(2) Share + learning (done/cap)"
    ,"③ share+冻（完成/上限）": "(3) Share + frozen (done/cap)"
    ,"下一臂": "Next arm"
    ,"拟合点预计还需": "Estimated remaining for the fit point"
    ,"自信 —": "Confidence —"
    ,"自动进行（默认开启）": "Automatic (on by default)"
    ,"停止报名": "Stop enrollment"
    ,"继续报名": "Resume enrollment"
    ,"刷新 share 拟合": "Refresh share-fit"
    ,"确认拟合并进入对比": "Confirm the fit and enter the contrast"
    ,"下载 Share 拟合曲线（含开场 q）": "Download the share-fit curve (including opening q)"
    ,"下载最新 Share 拟合权重（只读备份）": "Download the latest share-fit weights (read-only backup)"
    ,"「下载最新 Share 拟合权重」只读取已落盘的 prior 并传给浏览器，不改写、不移动服务器上的模型或编排状态。": "Download latest share-fit weights only reads a saved prior into the browser. It does not rewrite or move server models or orchestration state."
    ,"集体实验锁定": "Group-session lock"
    ,"锁定后：该时段内外部不能开始实验；预约格灰掉；已在进行中的受试者不受影响。 系统会生成一段召集密码，可在本机研究入口与隔离测试入口页底开启实验，便于用固定设备召集一批人。 本操作只写独立的锁定文件，不改 share 拟合、模型和当前持有者。": "While locked, outsiders cannot start; booking cells are grey; in-progress participants are unaffected. A convening password is generated so a local study entry and an isolated test entry can open the session at the page footer. This writes only an independent lock file."
    ,"锁定后：该时段内外部不能开始实验；预约格灰掉；已在进行中的受试者不受影响。 系统会生成一段召集密码，可在本机研究入口与隔离测试入口页底开启实验，便于用固定设备召集一批人。 本操作只写独立的锁定文件，不改 share 拟合、模型和当前持有者。": "While locked, outsiders cannot start; booking cells are grey; in-progress participants are unaffected. A convening password is generated so a local study entry and an isolated test entry can open the session at the page footer. This writes only an independent lock file."
    ,"开始（北京时间）": "Start (Beijing time)"
    ,"结束（北京时间）": "End (Beijing time)"
    ,"锁定并生成密码": "Lock and generate password"
    ,"解除锁定": "Unlock"
    ,"复制密码": "Copy password"
    ,"当前未锁定。": "Currently unlocked."
    ,"不可预约时间段": "Blocked booking windows"
    ,"按北京时间选择某一天的半小时块。保存后仅阻止本实例后续新预约；已有预约、进行中的实验、模型和全局互斥都不会被改写。": "Choose half-hour blocks for one day in Beijing time. Saving only blocks later bookings on this instance. Existing bookings, in-progress sessions, models, and the global mutex are not rewritten."
    ,"日期（北京时间）": "Date (Beijing time)"
    ,"当天全选": "Select the whole day"
    ,"当天清空勾选": "Clear the day's checks"
    ,"保存当天不可预约时段": "Save blocked windows for this day"
    ,"预约记录与四位凭证": "Bookings and four-digit codes"
    ,"实验 Run": "Study run"
    ,"参与者账号": "Participant accounts"
    ,"会话": "Sessions"
    ,"事件": "Events"
    ,"音频文件": "Audio files"
    ,"数据集发布": "Dataset release"
    ,"第一步": "Step 1"
    ,"初始化实验阶段账号": "Initialize stage accounts"
    ,"选择阶段后，系统自动锁定 cohort、数据来源和协议模式。所有真人实验默认记录研究必需数据；原始语音及后续共享范围由参与者的分层同意决定。登录码只显示一次。": "Choosing a stage locks cohort, data origin, and protocol mode. Live sessions record required research data by default; raw audio and later sharing follow layered consent. Login codes are shown once."
    ,"实验阶段": "Study stage"
    ,"指派人数": "Assigned count"
    ,"Run 标识（可选）": "Run id (optional)"
    ,"批次备注（可选）": "Batch note (optional)"
    ,"仅用于个人调试": "Personal debugging only"
    ,"生成账号与 ID": "Create accounts and IDs"
    ,"账号已生成": "Accounts created"
    ,"复制 TSV": "Copy TSV"
    ,"下载 CSV": "Download CSV"
    ,"请现在保存登录码；系统只保存散列，刷新后无法恢复完整登录码。": "Save the login codes now. The system stores only hashes and cannot restore full codes after refresh."
    ,"第二步": "Step 2"
    ,"查看记录和文件位置": "Inspect records and file locations"
    ,"存储路径": "Storage paths"
    ,"路径来自当前服务器配置。数据集发布需使用下方导出脚本执行同意范围和去标识检查。": "Paths come from the current server configuration. Dataset release must use the export script below for consent scope and de-identification checks."
    ,"Run 汇总": "Run summary"
    ,"账号初始化批次": "Account-initialization batches"
    ,"参与者实验记录": "Participant study records"
    ,"名单直接显示姓名、联系方式、身份、口音和只读实验状态（实验中 / 已完成 / 中途退出等）。点击“查看”读取该参与者的五窗口点击 / 打断、会话、质量检查、用户发言、智能体回答和问卷原文。": "The roster shows name, contact details, identity, accent, and read-only status. View opens five-window tap/interrupt counts, sessions, quality checks, user speech, agent replies, and questionnaire text."
    ,"参与者记录": "Participant record"
    ,"下载研究方归档副本": "Download the research-team archive copy"
    ,"关闭": "Close"
    ,"五个窗口的点击 / 打断": "Tap / interrupt counts in five windows"
    ,"事件类型": "Event type"
    ,"用户与智能体对话原文": "User and agent dialogue text"
    ,"每行对应一次实际发送给智能体的用户发言及其回答；“电脑实验系统”和“手机陪伴系统”分别标记。": "Each row is one user utterance actually sent to an agent and its reply. Computer study system and phone companion are labeled separately."
    ,"最近含文本事件原文": "Recent events that contain text"
    ,"问卷选择记录": "Questionnaire choice records"
    ,"显示受试者实际看到的题目文字与所选选项内容；内部稳定编码同时保留用于统计。": "Shows the question text the participant saw and the chosen option. Stable internal codes are kept for analysis."
    ,"问卷开放题原文": "Open-ended questionnaire text"
    ,"数据资产": "Data assets"
    ,"最近 200 条事件的类型分布": "Type distribution of the latest 200 events"
    ,"查看音频和模型文件路径": "Inspect audio and model file paths"
    ,"最近音频文件": "Recent audio files"
    ,"模型文件": "Model files"
    ,"正式实验门禁与 A→B 高级操作": "Formal-study gates and A→B advanced operations"
    ,"正式实验门禁": "Formal-study gates"
    ,"A→B 正式流水线": "Formal A→B pipeline"
    ,"正式 Study A/B 账号不由上方快捷表单创建。释放前服务端会检查完整会话、机会预算、模型快照、prior 哈希和 matched assignment。": "Formal Study A/B accounts are not created by the shortcut form above. Before release the server checks complete sessions, opportunity budgets, model snapshots, prior hashes, and matched assignments."
    ,"锁定 donor n": "Lock donor n"
    ,"锁定反馈上限 q": "Lock feedback cap q"
    ,"盲态饱和/精度决策记录": "Blinded saturation / precision decision record"
    ,"Reserve 释放原因": "Reserve-release reason"
    ,"读取状态": "Read status"
    ,"J(n,q) / Study B 分析": "J(n,q) / Study B analysis"
    ,"释放所选 Wave": "Release the selected wave"
    ,"冻结并准备 Study B": "Freeze and prepare Study B"
    ,"释放 Reserve": "Release reserve"
    ,"管理页数据视图": "Dashboard data view"
    ,"留空则自动生成": "Leave blank to auto-generate"
    ,"例如 chi-formal-20260715": "Example formal-20260715"
    ,"填写预注册判据的判定与分析记录位置": "Record the preregistered decision and analysis-record location"
    ,"记录实际 attrition 原因": "Record the actual attrition reason"
    ,"触摸反应延迟": "Touch reaction latency"
    ,"看到红色圆形后立即轻触，共 6 次": "Tap the red circle immediately when it appears, 6 times"
    ,"请稍后再来和我互动": "Come back to talk with me later"
    ,"若暂未开口，我将在 10 秒后主动打招呼": "If you have not spoken yet, I will greet you after 10 seconds"
    ,"请留在当前位置，持续与当前智能体自然对话；倾听回答时无需说话。": "Stay where you are and keep talking naturally with the current agent. You do not need to speak while listening."
    ,"当前实验编号": "Current study code"
  };

  const PATTERNS = [
    [/^编号尾号 (.+)$/, (_, id) => `Code ending ${id}`],
    [/^轻触我（还差 (\d+) 次）$/, (_, n) => `Tap me (${n} remaining)`],
    [/^点我（还差 (\d+) 次）$/, (_, n) => `Tap me (${n} remaining)`],
    [/^完成记录 · 编号尾号 (.+)$/, (_, id) => `Completion record · code ending ${id}`],
    [/^还需完成 (\d+) 项条件：$/, (_, n) => `${n} requirement(s) remaining:`],
    [/^连接前还需完成 (\d+) 项条件：$/, (_, n) => `${n} requirement(s) remaining before connecting:`],
    [/^第 (\d+)\/(\d+) 个六分钟统计块 · 前四分钟$/, (_, a, b) => `Six-minute block ${a}/${b} · first four minutes`],
    [/^第 (\d+)\/(\d+) 个六分钟统计块 · 后两分钟$/, (_, a, b) => `Six-minute block ${a}/${b} · final two minutes`],
    [/^第 (\d+)\/(\d+) 个六分钟统计块结束$/, (_, a, b) => `End of six-minute block ${a}/${b}`],
    [/^交流时段 (\d+)\/(\d+)$/, (_, a, b) => `Conversation period ${a}/${b}`],
    [/^你：(.+)$/, (_, value) => `You: ${value}`],
    [/^对话伙伴：(.+)$/, (_, value) => `Conversation partner: ${value}`],
    [/^陪伴伙伴：(.+)$/, (_, value) => `Phone companion: ${value}`],
    [/^手机连接已断开，(\d+) 秒后自动重连$/, (_, n) => `Phone disconnected; reconnecting in ${n} seconds`],
    [/^无法进入：.+$/, () => "Unable to enter the study. Check the participant code and try again."],
    [/^无法连接：.+$/, () => "Unable to connect. Check the participant code and network, then try again."],
    [/^提交失败：.+$/, () => "Submission failed. Please retry; contact the research team if the problem continues."],
    [/^话题本载入失败：.+$/, () => "The topic book could not be loaded. Please retry."],
    [/^话题保存失败：.+$/, () => "The topic could not be saved. Please retry."],
    [/^无法启动：.+$/, () => "The study could not start. Complete the displayed requirements and retry."]
    ,[/^本次记录不会标记为完整完成，请联系研究团队 · 编号尾号 (.+)$/, (_, id) => `This record will not be marked complete. Contact the research team · code ending ${id}`]
    ,[/^触摸反应中位数 (.+) ms · 范围 (.+)–(.+) ms$/, (_, median, low, high) => `Median touch reaction ${median} ms · range ${low}–${high} ms`]
    ,[/^点击反应中位数 (.+) ms · 范围 (.+)–(.+) ms$/, (_, median, low, high) => `Median tap reaction ${median} ms · range ${low}–${high} ms`]
    ,[/^中位 (.+) ms · P95 (.+) ms · 抖动 (.+) ms · 丢失 (.+)\/12$/, (_, median, p95, jitter, lost) => `Median ${median} ms · P95 ${p95} ms · jitter ${jitter} ms · lost ${lost}/12`]
    ,[/^正常说话音量可用 · SNR (.+) dB$/, (_, snr) => `Normal speaking volume is suitable · SNR ${snr} dB`]
    ,[/^检测未保存：.+$/, () => "The check was not saved. Please retry."]
    ,[/^检测失败：.+$/, () => "The check failed. Please retry."]
    ,[/^确定现在开始（(\d+)）$/, (_, seconds) => `Start now (${seconds})`]
    ,[/^进入全屏并开始实验（(\d+)）$/, (_, seconds) => `Enter fullscreen and start (${seconds})`]
    ,[/^未通过：(.+)$/, (_, detail) => `Did not pass: ${detail}`]
    ,[/^(.+)今日已用完 2 次机会，请联系研究团队。$/, (_, label) => `${EN[label] || label}You have used all 2 attempts for today. Contact the research team.`]
    ,[/^(.+)结果未能保存：(.+)$/, (_, label, detail) => `${EN[label] || label}Could not save result: ${detail}`]
    ,[/^(手机网络稳定性检测|手机麦克风与环境噪声检测|手机触摸反应延迟检测|手机点击反应延迟检测|手机浏览器与声音播放检测|手机浏览器与最低音量硬校准)(尚未完成|未通过)$/, (_, label, status) => `${EN[label] || label} ${status === "尚未完成" ? "not completed" : "did not pass"}`]
    ,[/^无法使用麦克风：.+$/, () => "The microphone could not be used. Check browser permission and retry."]
    ,[/^未能签署：.+$/, () => "The consent form could not be signed. Please retry."]
    ,[/^副本下载失败：.+$/, () => "The participant copy could not be downloaded. Please retry."]
    ,[/^同意已记录，但副本下载失败：.+$/, () => "Consent was recorded, but the participant copy could not be downloaded. Contact the principal researcher for a copy."]
    ,[/^准备失败：.+$/, () => "Phone preparation failed. Please retry."]
    ,[/^无法检测手机麦克风：.+$/, () => "The phone microphone could not be tested. Check microphone permission and retry."]
    ,[/^语音识别异常：.+$/, () => "Speech recognition failed. Please retry."]
    ,[/^对话失败：.+$/, () => "The conversation request failed. Please retry."]
    ,[/^服务端语音暂时不可用：.+$/, () => "Server speech is temporarily unavailable. The computer has paused; retry shortly."]
  ];

  function translate(value) {
    if (locale !== "en" || typeof value !== "string") return value;
    if (EN[value]) return EN[value];
    // HTML indentation and line breaks must not prevent an exact UI translation.
    const normalized = value.replace(/\s+/g, " ").trim();
    if (EN[normalized]) return EN[normalized];
    for (const [pattern, replacement] of PATTERNS) {
      if (pattern.test(normalized)) return normalized.replace(pattern, replacement);
    }
    return value;
  }

  const ATTRIBUTES = ["placeholder", "title", "aria-label", "alt"];
  const SKIP = 'script, style, textarea, [contenteditable="true"], [translate="no"], [data-i18n-skip]';

  /** Translate interface text only; opt-out containers preserve user content. */
  function translateTextNode(node) {
    if (!node.parentElement || node.parentElement.closest(SKIP)) return;
    const raw = node.nodeValue || "";
    const match = raw.match(/^(\s*)(.*?)(\s*)$/s);
    if (!match || !match[2]) return;
    const translated = translate(match[2]);
    if (translated !== match[2]) node.nodeValue = `${match[1]}${translated}${match[3]}`;
  }

  function apply(root = document) {
    if (locale !== "en") return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const parent = walker.currentNode.parentElement;
      if (!parent || parent.closest(SKIP)) continue;
      nodes.push(walker.currentNode);
    }
    nodes.forEach(translateTextNode);
    const selector = ATTRIBUTES.map((name) => `[${name}]`).join(",");
    // querySelectorAll excludes the root; newly inserted inputs also need labels.
    const elements = root.querySelectorAll ? [...root.querySelectorAll(selector)] : [];
    if (root.matches?.(selector)) elements.unshift(root);
    for (const element of elements) {
      if (element.closest(SKIP)) continue;
      for (const attribute of ATTRIBUTES) {
        const current = element.getAttribute(attribute);
        const translated = translate(current);
        // Write only changes: observing attributes otherwise causes a feedback loop.
        if (current && translated !== current) element.setAttribute(attribute, translated);
      }
    }
  }

  function localizedApiPath(path) {
    if (locale !== "en") return path;
    const url = new URL(path, location.origin);
    url.searchParams.set("lang", "en");
    return `${url.pathname}${url.search}`;
  }

  function setLanguage(next) {
    const normalized = next === "en" ? "en" : "zh-CN";
    localStorage.setItem(STORAGE_KEY, normalized);
    const url = new URL(location.href);
    url.searchParams.set("lang", normalized);
    location.assign(url.toString());
  }

  function bindLanguageButtons() {
    for (const button of document.querySelectorAll("[data-language-toggle]")) {
      button.textContent = locale === "en" ? "Chinese" : "English";
      button.setAttribute("aria-label", locale === "en" ? "Switch to Chinese" : "Switch to English");
      button.addEventListener("click", () => setLanguage(locale === "en" ? "zh-CN" : "en"));
    }
  }

  document.documentElement.lang = locale === "en" ? "en" : "zh-CN";
  window.TTA_I18N = {
    locale,
    isEnglish: locale === "en",
    t: translate,
    apply,
    apiPath: localizedApiPath,
    setLanguage
  };

  const ready = () => {
    bindLanguageButtons();
    apply(document);
    if (locale === "en") {
      const observer = new MutationObserver((records) => {
        for (const record of records) {
          if (record.type === "characterData") translateTextNode(record.target);
          if (record.type === "attributes") apply(record.target);
          for (const node of record.addedNodes) {
            if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
            else if (node.nodeType === Node.ELEMENT_NODE) apply(node);
          }
        }
      });
      observer.observe(document.body, {
        subtree: true, childList: true, characterData: true,
        attributes: true, attributeFilter: ATTRIBUTES
      });
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready, { once: true });
  else ready();
})();
