# **Security Policy**

## **Supported Versions**
We are committed to keeping the latest and most stable versions of **TelePrompt** secure and up-to-date. The following versions of the project are actively supported with security updates:

| Version   | Supported          |
| --------- | ------------------ |
| 5.x.x     | :white_check_mark: |
| 4.x.x     | :white_check_mark: |
| < 4.0     | :x:                |

**Note:** Older versions of the project (below version 4.0) are no longer supported and will not receive security updates. For the latest security fixes and features, please upgrade to the supported version.

---

## **Reporting a Vulnerability**
We take the security of **TelePrompt** seriously and encourage community members to report any potential vulnerabilities they discover. Here’s how to report a vulnerability effectively:

### **Steps to Report:**

1. **Private Reporting:**  
   - Please report security issues privately to our security team by sending an email to [security@saphyresolutions.com](mailto:security@saphyresolutions.com).  
   - We highly recommend **not** disclosing the vulnerability publicly until the issue is resolved to avoid exploitation.

2. **Vulnerability Details:**  
   When reporting, please provide the following information to help us address the issue quickly:
   - A clear description of the vulnerability and steps to reproduce it.
   - Any potential impact or exploitability details.
   - The version of **TelePrompt** you’re using.

3. **Follow-up Updates:**  
   - Once your report is received, you will get an acknowledgment within **48 hours**.
   - We will provide you with a status update within **7 business days** regarding the investigation and remediation efforts.

### **Vulnerability Resolution Process:**
- **Accepted Reports:** If the vulnerability is confirmed, we will prioritize a fix in the next release or patch and notify you when it’s resolved. We will also credit you (unless you wish to remain anonymous) for reporting the issue.
  
- **Declined Reports:** If we determine that the issue is not a security vulnerability, you will receive an explanation of why the issue was declined. We value feedback and encourage you to continue contributing to improving the project.

### **Responsible Disclosure:**
- All security vulnerabilities reported to us will be handled with care, and we will provide a timely and transparent resolution process. We are committed to ensuring that fixes are applied to production systems promptly.
  
### **Security Best Practices:**
- **Encryption:** Ensure that sensitive data such as API keys, authentication tokens, and passwords are encrypted both in transit and at rest.
- **Access Control:** Use role-based access controls (RBAC) for managing user permissions, ensuring that only authorized individuals have access to critical parts of the system.
- **Regular Audits:** We conduct regular security audits of the codebase, dependencies, and infrastructure to identify and fix potential vulnerabilities.
- **Dependencies:** We carefully track and update third-party dependencies to minimize risks from outdated or vulnerable libraries.
- **Environment Security:** Ensure that your environment variables (such as Google API keys) are securely stored and not hard-coded into the repository. Use secrets management tools wherever possible.

---

## **Additional Security Recommendations:**

1. **API Security:**  
   - Always use **API keys** securely by storing them in environment variables or encrypted configuration files, and never expose them in the repository.
   - Enforce **rate-limiting** on all external APIs to prevent abuse.

2. **Authentication and Authorization:**  
   - Utilize **OAuth** or **SSO (Single Sign-On)** for user authentication. Ensure that only authorized users can access sensitive areas of the platform.
  
3. **Code Review & Collaboration:**  
   - All code contributions are reviewed before being merged. Focus on detecting security flaws, reviewing third-party dependencies, and ensuring that no sensitive information is exposed in code.

---

**Thank you for helping us keep TelePrompt secure and effective. We are committed to transparency and swift action to ensure the safety of all users and contributors.**
