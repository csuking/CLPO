# CLPO Project Page Deployment Guide

This guide will help you deploy the CLPO project page to GitHub Pages, similar to the SvS project page.

## 📁 File Structure

```
CLPO/
├── docs/
│   ├── index.html          # Main project page
│   ├── _config.yml         # Jekyll configuration
│   └── DEPLOYMENT.md       # This file
├── CLPO-teaser.jpg        # Teaser image
├── CLPO-workflow.png      # Workflow diagram
└── README.md              # GitHub repository README
```

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add CLPO project page and documentation"
   git push origin main
   ```

2. **Ensure your images are in the right location:**
   - `CLPO-teaser.jpg` should be in the root directory
   - `CLPO-workflow.png` should be in the root directory
   - The HTML file references these with `../CLPO-teaser.jpg` and `../CLPO-workflow.png`

### Step 2: Enable GitHub Pages

1. Go to your GitHub repository settings
2. Scroll down to the "Pages" section
3. Under "Source", select "Deploy from a branch"
4. Choose "main" branch and "/docs" folder
5. Click "Save"

### Step 3: Configure Your Links

Update the following in your files:

1. **In `docs/index.html`:**
   - Replace `#` in the link buttons with actual URLs:
     ```html
     <a href="https://arxiv.org/abs/XXXX.XXXXX" class="link-btn">
         <i class="fas fa-file-alt"></i> Paper
     </a>
     <a href="https://github.com/your-username/CLPO" class="link-btn">
         <i class="fab fa-github"></i> Code
     </a>
     ```

2. **In `docs/_config.yml`:**
   - Replace `your-username` with your actual GitHub username
   - Update the URL to match your repository

3. **In `README.md`:**
   - Replace `your-username` with your actual GitHub username in clone commands

### Step 4: Update Content

1. **Replace placeholder information:**
   - Update author names in the citation
   - Add actual arXiv paper ID when available
   - Update dates in the news section
   - Add actual dataset and Twitter links

2. **Verify image paths:**
   - Make sure `CLPO-teaser.jpg` and `CLPO-workflow.png` are accessible
   - Test the image links in your deployed page

### Step 5: Custom Domain (Optional)

If you want a custom domain like `clpo.ai`:

1. Add a `CNAME` file in the `/docs` directory:
   ```bash
   echo "clpo.ai" > docs/CNAME
   ```

2. Configure your domain's DNS settings to point to GitHub Pages
3. Update the `url` in `_config.yml` to your custom domain

## 🔧 Customization Options

### Styling
- Modify the CSS in `docs/index.html` to match your preferred colors/fonts
- The current design uses a purple gradient theme similar to SvS

### Content Sections
- Add more sections by copying the section template
- Update the results table with your actual experimental data
- Add more feature cards or modify existing ones

### Interactive Elements
- Add JavaScript for animations or interactive charts
- Include demo videos or interactive examples

## 📱 Mobile Responsiveness

The page is designed to be mobile-responsive with:
- Flexible grid layouts
- Responsive typography
- Mobile-friendly navigation
- Optimized image sizing

## 🔍 SEO Optimization

The page includes:
- Proper meta tags
- Semantic HTML structure
- Alt text for images
- Structured data for better search engine indexing

## 🚨 Troubleshooting

### Common Issues:

1. **Images not loading:**
   - Check file paths in HTML
   - Ensure images are committed to the repository
   - Verify image file names match exactly (case-sensitive)

2. **Page not updating:**
   - GitHub Pages can take a few minutes to deploy
   - Check the Actions tab for build status
   - Clear your browser cache

3. **Styling issues:**
   - Verify CSS syntax in the HTML file
   - Check for any JavaScript errors in browser console

### Getting Help:

- Check GitHub Pages documentation
- Review the SvS repository structure for reference
- Test locally by opening `docs/index.html` in a browser

## 📊 Analytics (Optional)

To track page visits, add Google Analytics:

1. Get a Google Analytics tracking ID
2. Add the tracking code to the `<head>` section of `index.html`
3. Monitor your page performance and visitor statistics

## 🎯 Next Steps

After deployment:
1. Share your project page URL
2. Update social media links
3. Add the page URL to your paper submission
4. Consider adding more interactive elements or demos

Your CLPO project page will be available at:
`https://your-username.github.io/CLPO/`

## 📞 Support

If you encounter any issues with deployment, please:
1. Check the GitHub Pages documentation
2. Review the repository settings
3. Ensure all files are properly committed and pushed

Good luck with your CLPO project page! 🚀
