// API Configuration
const API_URL = 'http://localhost:8002';
console.log('API URL:', API_URL);

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Financial News Analysis application initialized');
    
    // Check API connection
    checkApiConnection();
    
    // Setup navigation
    setupNavigation();
    
    // Setup news form
    setupNewsForm();
    
    // Setup batch form
    setupBatchForm();
    
    // Setup stock search
    setupStockSearch();
    
    // Load rankings
    loadRankings();
});

// Check API connection
async function checkApiConnection() {
    console.log('Checking API connection...');
    const startTime = performance.now();
    
    try {
        const response = await fetch(`${API_URL}/`);
        const endTime = performance.now();
        
        if (response.ok) {
            console.log(`API connection successful (${Math.round(endTime - startTime)}ms)`);
            return true;
        } else {
            console.error('API connection failed:', response.status, response.statusText);
            showApiWarning();
            return false;
        }
    } catch (error) {
        console.error('API connection error:', error);
        showApiWarning();
        return false;
    }
}

// Show API warning
function showApiWarning() {
    // Create warning element
    const warningEl = document.createElement('div');
    warningEl.className = 'api-warning';
    warningEl.innerHTML = `
        <div class="api-warning-content">
            <strong>Warning:</strong> Cannot connect to API server. Some features may not work.
            <button class="api-warning-close">&times;</button>
        </div>
    `;
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .api-warning {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: #f39c12;
            color: white;
            padding: 0.5rem;
            text-align: center;
            z-index: 1000;
        }
        .api-warning-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
        }
        .api-warning-close {
            background: none;
            border: none;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
        }
    `;
    
    // Add to document
    document.head.appendChild(style);
    document.body.prepend(warningEl);
    
    // Add close button event
    warningEl.querySelector('.api-warning-close').addEventListener('click', function() {
        warningEl.remove();
    });
}

// Setup navigation
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get section ID
            const sectionId = this.getAttribute('data-section');
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show active section
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                section.classList.remove('active');
            });
            document.getElementById(sectionId).classList.add('active');
            
            // Update URL hash
            window.location.hash = sectionId;
        });
    });
    
    // Handle initial hash
    if (window.location.hash) {
        const hash = window.location.hash.substring(1);
        const link = document.querySelector(`.nav-link[data-section="${hash}"]`);
        if (link) {
            link.click();
        }
    }
}

// Setup news form
function setupNewsForm() {
    const newsForm = document.getElementById('news-form');
    
    if (newsForm) {
        newsForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(newsForm);
            const newsItem = {
                title: formData.get('title'),
                content: formData.get('content'),
                symbol: formData.get('symbol') || null,
                date: formData.get('date') || null,
                source: formData.get('source') || null
            };
            
            // Analyze news
            await analyzeNews(newsItem);
        });
    }
}

// Setup batch form
function setupBatchForm() {
    const batchForm = document.getElementById('batch-form');
    const batchFileInput = document.getElementById('batch-file');
    const fileInfo = document.querySelector('.file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const submitBtn = batchForm.querySelector('button[type="submit"]');
    const downloadTemplateBtn = document.getElementById('download-template');
    const exportResultsBtn = document.getElementById('export-results');
    
    if (batchFileInput) {
        batchFileInput.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                const file = this.files[0];
                
                // Validate file type
                if (!file.name.endsWith('.csv')) {
                    alert('Please select a CSV file.');
                    this.value = '';
                    return;
                }
                
                // Validate file size (max 5MB)
                if (file.size > 5 * 1024 * 1024) {
                    alert('File size exceeds 5MB limit.');
                    this.value = '';
                    return;
                }
                
                // Show file info
                fileName.textContent = file.name;
                fileInfo.classList.remove('hidden');
                submitBtn.disabled = false;
            } else {
                fileInfo.classList.add('hidden');
                submitBtn.disabled = true;
            }
        });
    }
    
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', function() {
            batchFileInput.value = '';
            fileInfo.classList.add('hidden');
            submitBtn.disabled = true;
        });
    }
    
    if (downloadTemplateBtn) {
        downloadTemplateBtn.addEventListener('click', function(e) {
            e.preventDefault();
            downloadCsvTemplate();
        });
    }
    
    if (exportResultsBtn) {
        exportResultsBtn.addEventListener('click', function(e) {
            e.preventDefault();
            exportResultsAsCsv();
        });
    }
    
    if (batchForm) {
        batchForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (batchFileInput.files.length === 0) {
                alert('Please select a CSV file.');
                return;
            }
            
            await handleBatchSubmit(batchFileInput.files[0]);
        });
    }
}

// Process batch file
function processBatchFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const content = e.target.result;
            const lines = content.split('\n');
            
            // Check if file is empty
            if (lines.length <= 1) {
                reject('CSV file is empty or invalid.');
                return;
            }
            
            // Parse header
            const header = lines[0].split(',').map(h => h.trim().toLowerCase());
            
            // Validate required columns
            if (!header.includes('title') || !header.includes('content')) {
                reject('CSV file must contain "title" and "content" columns.');
                return;
            }
            
            // Parse rows
            const newsItems = [];
            const invalidLines = [];
            
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                
                try {
                    // Split by comma, but respect quotes
                    const values = line.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g) || [];
                    
                    // Remove quotes from values
                    const cleanValues = values.map(v => v.replace(/^"|"$/g, '').trim());
                    
                    // Create news item object
                    const newsItem = {};
                    header.forEach((col, index) => {
                        if (index < cleanValues.length) {
                            newsItem[col] = cleanValues[index] || null;
                        }
                    });
                    
                    // Validate required fields
                    if (newsItem.title && newsItem.content) {
                        newsItems.push(newsItem);
                    } else {
                        invalidLines.push(i + 1);
                    }
                } catch (error) {
                    invalidLines.push(i + 1);
                }
            }
            
            resolve({
                newsItems,
                invalidLines,
                totalLines: lines.length - 1
            });
        };
        
        reader.onerror = function() {
            reject('Error reading file.');
        };
        
        reader.readAsText(file);
    });
}

// Handle batch submit
async function handleBatchSubmit(file) {
    showLoading('Processing CSV file...');
    
    try {
        // Process file
        const { newsItems, invalidLines, totalLines } = await processBatchFile(file);
        
        if (newsItems.length === 0) {
            hideLoading();
            alert('No valid news items found in the CSV file.');
            return;
        }
        
        // Show batch results container
        const batchResults = document.getElementById('batch-results');
        batchResults.classList.remove('hidden');
        
        // Update summary
        document.getElementById('total-articles').textContent = totalLines;
        document.getElementById('processed-articles').textContent = newsItems.length;
        document.getElementById('success-rate').textContent = 
            `${Math.round((newsItems.length / totalLines) * 100)}%`;
        
        // Process in batches to avoid overwhelming the API
        const batchSize = 5;
        const results = [];
        const tableBody = document.querySelector('#batch-table tbody');
        tableBody.innerHTML = '';
        
        for (let i = 0; i < newsItems.length; i += batchSize) {
            const batch = newsItems.slice(i, i + batchSize);
            
            showLoading(`Processing items ${i+1} to ${Math.min(i+batchSize, newsItems.length)} of ${newsItems.length}...`);
            
            try {
                const batchResults = await fetch(`${API_URL}/batch-analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(batch)
                }).then(res => res.json());
                
                // Add results
                for (let j = 0; j < batch.length; j++) {
                    const item = batch[j];
                    const analysis = batchResults[j];
                    
                    results.push({
                        title: item.title,
                        content: item.content,
                        sentiment: analysis.sentiment,
                        sentiment_probability: analysis.sentiment_probability,
                        impact: analysis.impact,
                        impact_probability: analysis.impact_probability,
                        symbols: analysis.extracted_symbols.join(', '),
                        date: item.date || '',
                        source: item.source || ''
                    });
                    
                    // Add row to table
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${item.title}</td>
                        <td><span class="badge ${getSentimentClass(analysis.sentiment)}">${analysis.sentiment}</span></td>
                        <td><span class="badge ${getImpactClass(analysis.impact)}">${analysis.impact}</span></td>
                        <td>${analysis.extracted_symbols.join(', ')}</td>
                        <td>${item.date || '-'}</td>
                        <td>${item.source || '-'}</td>
                    `;
                    tableBody.appendChild(row);
                }
            } catch (error) {
                console.error('Error processing batch:', error);
            }
        }
        
        // Scroll to results
        batchResults.scrollIntoView({ behavior: 'smooth' });
        
        // Store results for export
        window.batchResults = results;
        
        // Show warning for invalid lines
        if (invalidLines.length > 0) {
            alert(`Warning: ${invalidLines.length} lines were invalid and skipped. (Lines: ${invalidLines.join(', ')})`);
        }
    } catch (error) {
        console.error('Error processing batch file:', error);
        alert(`Error: ${error}`);
    } finally {
        hideLoading();
    }
}

// Download CSV template
function downloadCsvTemplate() {
    const template = `title,content,symbol,date,source
"Apple Reports Strong Earnings","Apple reports record-breaking quarterly revenue of $123.9 billion, exceeding analyst expectations.",AAPL,2025-03-15,Bloomberg
"Tesla Announces New Model","Tesla unveils new electric vehicle model with 500 mile range and advanced self-driving capabilities.",TSLA,2025-03-14,Reuters
"Microsoft Cloud Growth","Microsoft reports 32% growth in cloud services revenue, driving overall company performance.",MSFT,2025-03-13,CNBC`;
    
    const blob = new Blob([template], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'news_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Export results as CSV
function exportResultsAsCsv() {
    if (!window.batchResults || window.batchResults.length === 0) {
        alert('No results to export.');
        return;
    }
    
    // Create CSV content
    const headers = Object.keys(window.batchResults[0]).join(',');
    const rows = window.batchResults.map(item => {
        return Object.values(item).map(value => {
            // Wrap values in quotes and escape existing quotes
            return `"${String(value).replace(/"/g, '""')}"`;
        }).join(',');
    });
    
    const csvContent = [headers, ...rows].join('\n');
    
    // Download file
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `news_analysis_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Setup stock search
function setupStockSearch() {
    const searchBtn = document.getElementById('search-stock');
    const symbolInput = document.getElementById('stock-symbol');
    
    if (searchBtn && symbolInput) {
        searchBtn.addEventListener('click', async function() {
            const symbol = symbolInput.value.trim();
            
            if (!symbol) {
                alert('Please enter a stock symbol.');
                return;
            }
            
            await getStockDetails(symbol);
        });
        
        symbolInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                searchBtn.click();
            }
        });
    }
}

// Analyze news
async function analyzeNews(newsItem) {
    showLoading();
    
    console.log('Analyzing news:', newsItem);
    const startTime = performance.now();
    
    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newsItem)
        });
        
        const endTime = performance.now();
        console.log(`API response time: ${Math.round(endTime - startTime)}ms`);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const analysis = await response.json();
        console.log('Analysis result:', analysis);
        
        // Display results
        displayAnalysisResults(analysis);
    } catch (error) {
        console.error('Error analyzing news:', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Get stock rankings
async function getStockRankings() {
    try {
        const response = await fetch(`${API_URL}/rankings`);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error getting stock rankings:', error);
        return null;
    }
}

// Get stock details
async function getStockDetails(symbol) {
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/rankings/${symbol}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                alert(`Stock symbol "${symbol}" not found.`);
            } else {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            return;
        }
        
        const stock = await response.json();
        displayStockDetails(stock);
    } catch (error) {
        console.error('Error getting stock details:', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display analysis results
function displayAnalysisResults(analysis) {
    const resultsContainer = document.getElementById('analysis-results');
    
    if (resultsContainer) {
        // Update sentiment
        const sentimentBadge = document.getElementById('sentiment-badge');
        const sentimentConfidence = document.getElementById('sentiment-confidence');
        
        sentimentBadge.textContent = analysis.sentiment;
        sentimentBadge.className = `badge ${getSentimentClass(analysis.sentiment)}`;
        sentimentConfidence.textContent = `${Math.round(analysis.sentiment_probability * 100)}%`;
        
        // Update impact
        const impactBadge = document.getElementById('impact-badge');
        const impactConfidence = document.getElementById('impact-confidence');
        
        impactBadge.textContent = analysis.impact;
        impactBadge.className = `badge ${getImpactClass(analysis.impact)}`;
        impactConfidence.textContent = `${Math.round(analysis.impact_probability * 100)}%`;
        
        // Update symbols
        const symbolsContainer = document.getElementById('symbols-container');
        
        if (analysis.extracted_symbols.length > 0) {
            symbolsContainer.textContent = analysis.extracted_symbols.join(', ');
        } else {
            symbolsContainer.textContent = 'No symbols detected';
        }
        
        // Show results
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
}

// Load rankings
async function loadRankings() {
    showLoading();
    
    try {
        const rankings = await getStockRankings();
        
        if (rankings) {
            displayRankings(rankings);
        }
    } catch (error) {
        console.error('Error loading rankings:', error);
    } finally {
        hideLoading();
    }
}

// Display rankings
function displayRankings(rankings) {
    const tableBody = document.querySelector('#rankings-table tbody');
    
    if (tableBody) {
        tableBody.innerHTML = '';
        
        rankings.ranked_stocks.forEach((stock, index) => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${stock.symbol}</td>
                <td>${stock.ranking_score.toFixed(2)}</td>
                <td><span class="badge ${getSentimentClass(getSentimentLabel(stock.overall_sentiment_mean))}">${Math.round(stock.positive_ratio * 100)}%</span></td>
                <td>${stock.impact_score_mean.toFixed(2)}</td>
                <td>${stock.news_count}</td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Update chart
        updateRankingsChart(rankings.ranked_stocks);
        
        // Setup refresh button
        const refreshBtn = document.getElementById('refresh-rankings');
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', loadRankings);
        }
    }
}

// Display stock details
function displayStockDetails(stock) {
    const detailsContainer = document.getElementById('stock-details');
    
    if (detailsContainer) {
        // Update stock name
        document.getElementById('stock-name').textContent = stock.symbol;
        
        // Update sentiment badge
        const sentimentBadge = document.getElementById('stock-sentiment-badge');
        const sentimentLabel = getSentimentLabel(stock.overall_sentiment_mean);
        
        sentimentBadge.textContent = sentimentLabel;
        sentimentBadge.className = `badge ${getSentimentClass(sentimentLabel)}`;
        
        // Update metrics
        document.getElementById('stock-score').textContent = stock.ranking_score.toFixed(2);
        document.getElementById('stock-impact').textContent = stock.impact_score_mean.toFixed(2);
        document.getElementById('stock-news-count').textContent = stock.news_count;
        
        // Update chart
        updateSentimentChart(stock);
        
        // Show details
        detailsContainer.classList.remove('hidden');
    }
}

// Update rankings chart
function updateRankingsChart(stocks) {
    const chartCanvas = document.getElementById('rankings-chart');
    
    if (chartCanvas) {
        // Sort stocks by ranking score
        const sortedStocks = [...stocks].sort((a, b) => b.ranking_score - a.ranking_score);
        
        // Get top 10 stocks
        const topStocks = sortedStocks.slice(0, 10);
        
        // Prepare data
        const labels = topStocks.map(stock => stock.symbol);
        const data = topStocks.map(stock => stock.ranking_score);
        
        // Destroy existing chart if it exists
        if (window.rankingsChart) {
            window.rankingsChart.destroy();
        }
        
        // Create new chart
        window.rankingsChart = new Chart(chartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Ranking Score',
                    data: data,
                    backgroundColor: '#4CAF50',
                    borderColor: '#388E3C',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Ranking Score'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Stock Symbol'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const stock = topStocks[context.dataIndex];
                                return [
                                    `Score: ${stock.ranking_score.toFixed(2)}`,
                                    `Sentiment: ${Math.round(stock.positive_ratio * 100)}% positive`,
                                    `Impact: ${stock.impact_score_mean.toFixed(2)}`,
                                    `News Count: ${stock.news_count}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update sentiment chart
function updateSentimentChart(stock) {
    const chartCanvas = document.getElementById('sentiment-chart');
    
    if (chartCanvas) {
        // Prepare data
        const data = [
            stock.positive_ratio,
            stock.negative_ratio,
            stock.neutral_ratio
        ];
        
        // Destroy existing chart if it exists
        if (window.sentimentChart) {
            window.sentimentChart.destroy();
        }
        
        // Create new chart
        window.sentimentChart = new Chart(chartCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#4CAF50',
                        '#F44336',
                        '#9E9E9E'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const percentage = Math.round(value * 100);
                                return `${context.label}: ${percentage}%`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Show loading overlay
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = document.getElementById('loading-message');
    
    if (messageEl) {
        messageEl.textContent = message;
    }
    
    if (overlay) {
        overlay.classList.remove('hidden');
    }
}

// Hide loading overlay
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

// Get sentiment class
function getSentimentClass(sentiment) {
    sentiment = sentiment.toUpperCase();
    
    if (sentiment === 'POSITIVE') {
        return 'positive';
    } else if (sentiment === 'NEGATIVE') {
        return 'negative';
    } else {
        return 'neutral';
    }
}

// Get impact class
function getImpactClass(impact) {
    impact = impact.toLowerCase();
    
    if (impact.includes('high')) {
        return 'high';
    } else if (impact.includes('low')) {
        return 'low';
    } else {
        return 'medium';
    }
}

// Get sentiment label from score
function getSentimentLabel(score) {
    if (score >= 0.6) {
        return 'POSITIVE';
    } else if (score <= 0.4) {
        return 'NEGATIVE';
    } else {
        return 'NEUTRAL';
    }
}