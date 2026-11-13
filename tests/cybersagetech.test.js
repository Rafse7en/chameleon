import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("CyberSage Technologies", () => {
  it("should load the homepage and view cybersecurity services", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({ url: "https://cybersagetech.com" });

    const homeHeadingResult = await testdriver.assert(
      "the CyberSage Technologies headline 'Defending Your Digital World' is visible"
    );
    expect(homeHeadingResult).toBeTruthy();

    const servicesLink = await testdriver.find("Services link in top navigation bar");
    await servicesLink.click();

    await testdriver.find("Cybersecurity Services heading", { timeout: 15000 });

    const servicesHeadingResult = await testdriver.assert(
      "the heading 'Cybersecurity Services' is visible"
    );
    expect(servicesHeadingResult).toBeTruthy();
  });

  it("should open the contact page and enter contact name", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({ url: "https://cybersagetech.com/contact.html" });

    const nameInput = await testdriver.find("Full Name input field", { timeout: 15000 });
    await nameInput.click();
    await testdriver.type("Test User");

    const nameFilledResult = await testdriver.assert(
      "'Test User' is entered in the Full Name field"
    );
    expect(nameFilledResult).toBeTruthy();
  });
});
