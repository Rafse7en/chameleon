import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("CyberSage Technologies", () => {
  it("should navigate to services and fill out contact form", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({ url: "https://cybersagetech.com" });

    const assertResult = await testdriver.assert("The CyberSage Technologies homepage is visible with headline Defending Your Digital World");
    expect(assertResult).toBeTruthy();

    const servicesLink = await testdriver.find("Services navigation link in header");
    await servicesLink.click();

    await testdriver.find("Cybersecurity Services heading", { timeout: 15000 });
    const servicesAssertResult = await testdriver.assert("The Cybersecurity Services page is displayed with heading Cybersecurity Services");
    expect(servicesAssertResult).toBeTruthy();

    const contactLink = await testdriver.find("Contact Us navigation link in header");
    await contactLink.click();

    await testdriver.find("Get In Touch heading", { timeout: 15000 });
    const contactAssertResult = await testdriver.assert("The Contact Us page is displayed with heading Get In Touch and the contact form");
    expect(contactAssertResult).toBeTruthy();

    const nameInput = await testdriver.find("Full Name input field");
    await nameInput.click();
    await testdriver.type("Test User");

    await testdriver.pressKeys(["Tab"]);
    await testdriver.type("test@example.com");

    await testdriver.pressKeys(["Tab"]);
    await testdriver.type("Test Company");

    const formAssertResult = await testdriver.assert("The contact form fields Full Name, Work Email, and Organization are filled in");
    expect(formAssertResult).toBeTruthy();
  });
});
