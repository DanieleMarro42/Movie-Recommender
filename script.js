document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('movieSearch');
    const searchResults = document.getElementById('searchResults');
    const searchBtn = document.getElementById('searchBtn');

    const resultSection = document.getElementById('resultSection');
    const selectedTitleFn = document.getElementById('selectedTitle');
    const selectedClusterFn = document.getElementById('selectedCluster');
    const selectedOverviewFn = document.getElementById('selectedOverview');
    const recommendationsGrid = document.getElementById('recommendationsGrid');
    const moviePosterFn = document.getElementById('moviePoster');
    const baseUrl = "https://image.tmdb.org/t/p/w500";

    let debounceTimer;

    // [FIX] Show dropdown again when focusing input if text exists
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.trim().length >= 2 && searchResults.children.length > 0) {
            searchResults.classList.remove('hidden');
        }
    });

    // --- Typeahead Search ---
    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const query = e.target.value.trim();

        if (query.length < 2) {
            searchResults.classList.add('hidden');
            return;
        }

        // [FIX] Ensure dropdown is visible when typing starts
        searchResults.classList.remove('hidden');

        debounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
                const data = await response.json();
                renderSearchResults(data);
            } catch (err) {
                console.error("Search error:", err);
            }
        }, 300);
    });

    // --- Render Search Dropdown ---
    function renderSearchResults(results) {
        searchResults.innerHTML = '';
        if (results.length === 0) {
            searchResults.classList.add('hidden');
            return;
        }

        results.forEach(movie => {
            const div = document.createElement('div');
            div.className = 'search-result-item';
            div.innerHTML = `
                <div class="search-result-content">
                    <img src="${movie.poster_path ? baseUrl + movie.poster_path : 'https://via.placeholder.com/45x68?text=X'}" class="search-thumb">
                    <div class="search-text">
                        <span class="search-title">${movie.title}</span>
                        <span class="result-badge">Cluster ${movie.cluster}</span>
                    </div>
                </div>
            `;
            div.addEventListener('click', () => {
                searchInput.value = movie.title;
                searchResults.classList.add('hidden');
                fetchRecommendations(movie.title);
            });
            searchResults.appendChild(div);
        });

        searchResults.classList.remove('hidden');
    }

    // --- Search Button Click ---
    searchBtn.addEventListener('click', () => {
        const title = searchInput.value.trim();
        // Force hide dropdown
        searchResults.classList.add('hidden');
        if (title) {
            fetchRecommendations(title);
        }
    });

    // --- Fetch Recommendations ---
    async function fetchRecommendations(title) {
        // UI Loading State could be added here
        searchBtn.textContent = '...';

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: title })
            });

            const data = await response.json();

            if (response.ok) {
                updateUI(data);
            } else {
                alert(data.error || 'Something went wrong.');
            }
        } catch (err) {
            console.error(err);
            alert("Failed to connect to server.");
        } finally {
            searchBtn.textContent = 'Analyze';
        }
    }

    // --- Update UI with Results ---
    function updateUI(data) {
        // [FIX] Ensure dropdown is hidden when results are shown
        searchResults.classList.add('hidden');

        // Update Main Card
        selectedTitleFn.textContent = data.title;
        selectedClusterFn.textContent = data.cluster;
        selectedOverviewFn.textContent = data.overview;

        // Update Poster
        if (data.poster_path) {
            moviePosterFn.src = baseUrl + data.poster_path;
            moviePosterFn.classList.remove('hidden');
        } else {
            moviePosterFn.src = "";
            moviePosterFn.classList.add('hidden');
        }

        // Clear existing grid
        recommendationsGrid.innerHTML = '';

        // Animate section in
        resultSection.classList.remove('hidden');
        resultSection.classList.add('fade-in');

        // Populate Recommendations
        data.recommendations.forEach((rec, index) => {
            const card = document.createElement('div');
            card.className = 'rec-card fade-in';
            card.style.animationDelay = `${index * 0.1}s`; // Stagger animation

            const posterUrl = rec.poster_path ? baseUrl + rec.poster_path : 'https://via.placeholder.com/500x750?text=No+Image';

            card.innerHTML = `
                <img src="${posterUrl}" alt="${rec.title}" class="rec-poster">
                <div class="rec-content">
                    <h3 class="rec-title">${rec.title}</h3>
                    <span class="rec-cluster">Cluster Group: ${rec.cluster}</span>
                    <p class="rec-overview">${rec.overview}</p>
                </div>
            `;

            // Allow clicking a recommendation to research it
            card.addEventListener('click', () => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
                searchInput.value = rec.title;
                fetchRecommendations(rec.title);
            });

            recommendationsGrid.appendChild(card);
        });
    }

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        // [FIX] Logic to close dropdown if clicking anywhere else
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.add('hidden');
        }
    });
});
